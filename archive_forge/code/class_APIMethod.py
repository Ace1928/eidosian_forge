from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from apitools.base.py import  exceptions as apitools_exc
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis import apis_map
import six
class APIMethod(object):
    """A data holder for method information for an API collection."""

    def __init__(self, service, name, api_collection, method_config, disable_pagination=False):
        self._service = service
        self._method_name = name
        self._disable_pagination = disable_pagination
        self.collection = api_collection
        self.name = method_config.method_id
        dotted_path = self.collection.full_name + NAME_SEPARATOR
        if self.name.startswith(dotted_path):
            self.name = self.name[len(dotted_path):]
        self.path = _RemoveVersionPrefix(self.collection.api_version, method_config.relative_path)
        self.params = method_config.ordered_params
        if method_config.flat_path:
            self.detailed_path = _RemoveVersionPrefix(self.collection.api_version, method_config.flat_path)
            self.detailed_params = resource.GetParamsFromPath(method_config.flat_path)
        else:
            self.detailed_path = self.path
            self.detailed_params = self.params
        self.http_method = method_config.http_method
        self.request_field = method_config.request_field
        self.request_type = method_config.request_type_name
        self.response_type = method_config.response_type_name
        self._request_collection = self._RequestCollection()
        self.query_params = method_config.query_params

    @property
    def resource_argument_collection(self):
        """Gets the collection that should be used to represent the resource.

    Most of the time this is the same as request_collection because all methods
    in a collection operate on the same resource and so the API method takes
    the same parameters that make up the resource.

    One exception is List methods where the API parameters are for the parent
    collection. Because people don't specify the resource directly for list
    commands this also returns the parent collection for parsing purposes.

    The other exception is Create methods. They reference the parent collection
    list Like, but the difference is that we *do* want to specify the actual
    resource on the command line, so the original resource collection is
    returned here instead of the one that matches the API methods. When
    generating the request, you must figure out how to generate the message
    correctly from the parsed resource (as you cannot simply pass the reference
    to the API).

    Returns:
      APICollection: The collection.
    """
        if self.IsList():
            return self._request_collection
        return self.collection

    @property
    def request_collection(self):
        """Gets the API collection that matches the parameters of the API method."""
        return self._request_collection

    def GetRequestType(self):
        """Gets the apitools request class for this method."""
        return self._service.GetRequestType(self._method_name)

    def GetResponseType(self):
        """Gets the apitools response class for this method."""
        return self._service.GetResponseType(self._method_name)

    def GetEffectiveResponseType(self):
        """Gets the effective apitools response class for this method.

    This will be different from GetResponseType for List methods if we are
    extracting the list of response items from the overall response. This will
    always match the type of response that Call() returns.

    Returns:
      The apitools Message object.
    """
        if (item_field := self.ListItemField()) and self.HasTokenizedRequest():
            return arg_utils.GetFieldFromMessage(self.GetResponseType(), item_field).type
        else:
            return self.GetResponseType()

    def GetMessageByName(self, name):
        """Gets a arbitrary apitools message class by name.

    This method can be used to get arbitrary apitools messages from the
    underlying service. Examples:

    policy_type = method.GetMessageByName('Policy')
    status_type = method.GetMessageByName('Status')

    Args:
      name: str, the name of the message to return.
    Returns:
      The apitools Message object.
    """
        msgs = self._service.client.MESSAGES_MODULE
        return getattr(msgs, name, None)

    def IsList(self):
        """Determines whether this is a List method."""
        return self._method_name == 'List'

    def HasTokenizedRequest(self):
        """Determines whether this is a method that supports paging."""
        return not self._disable_pagination and 'pageToken' in self._RequestFieldNames() and ('nextPageToken' in self._ResponseFieldNames())

    def BatchPageSizeField(self):
        """Gets the name of the page size field in the request if it exists."""
        request_fields = self._RequestFieldNames()
        if 'maxResults' in request_fields:
            return 'maxResults'
        if 'pageSize' in request_fields:
            return 'pageSize'
        return None

    def ListItemField(self):
        """Gets the name of the field that contains the items in paginated response.

    This will return None if the method is not a paginated or if a single
    repeated field of items could not be found in the response type.

    Returns:
      str, The name of the field or None.
    """
        if self._disable_pagination:
            return None
        response = self.GetResponseType()
        found = [f for f in response.all_fields() if f.variant == messages.Variant.MESSAGE and f.repeated]
        if len(found) == 1:
            return found[0].name
        else:
            return None

    def _RequestCollection(self):
        """Gets the collection that matches the API parameters of this method.

    Methods apply to elements of a collection. The resource argument is always
    of the type of that collection.  List is an exception where you are listing
    items of that collection so the argument to be provided is that of the
    parent collection. This method returns the collection that should be used
    to parse the resource for this specific method.

    Returns:
      APICollection, The collection to use or None if no parent collection could
      be found.
    """
        if self.detailed_params == self.collection.detailed_params:
            return self.collection
        collections = GetAPICollections(self.collection.api_name, self.collection.api_version)
        for c in collections:
            if self.detailed_params == c.detailed_params:
                return c
        return None

    def _RequestFieldNames(self):
        """Gets the fields that are actually a part of the request message.

    For APIs that use atomic names, this will only be the single name parameter
    (and any other message fields) but not the detailed parameters.

    Returns:
      [str], The field names.
    """
        return [f.name for f in self.GetRequestType().all_fields()]

    def _ResponseFieldNames(self):
        """Gets the fields that are actually a part of the response message.

    Returns:
      [str], The field names.
    """
        return [f.name for f in self.GetResponseType().all_fields()]

    def Call(self, request, client=None, global_params=None, raw=False, limit=None, page_size=None):
        """Executes this method with the given arguments.

    Args:
      request: The apitools request object to send.
      client: base_api.BaseApiClient, An API client to use for making requests.
      global_params: {str: str}, A dictionary of global parameters to send with
        the request.
      raw: bool, True to not do any processing of the response, False to maybe
        do processing for List results.
      limit: int, The max number of items to return if this is a List method.
      page_size: int, The max number of items to return in a page if this API
        supports paging.

    Returns:
      The response from the API.
    """
        if client is None:
            client = apis.GetClientInstance(self.collection.api_name, self.collection.api_version)
        service = _GetService(client, self.collection.name)
        request_func = self._GetRequestFunc(service, request, raw=raw, limit=limit, page_size=page_size)
        try:
            return request_func(global_params=global_params)
        except apitools_exc.InvalidUserInputError as e:
            log.debug('', exc_info=True)
            raise APICallError(str(e))

    def _GetRequestFunc(self, service, request, raw=False, limit=None, page_size=None):
        """Gets a request function to call and process the results.

    If this is a method with paginated response, it may flatten the response
    depending on if the List Pager can be used.

    Args:
      service: The apitools service that will be making the request.
      request: The apitools request object to send.
      raw: bool, True to not do any processing of the response, False to maybe
        do processing for List results.
      limit: int, The max number of items to return if this is a List method.
      page_size: int, The max number of items to return in a page if this API
        supports paging.

    Returns:
      A function to make the request.
    """
        if raw or self._disable_pagination:
            return self._NormalRequest(service, request)
        item_field = self.ListItemField()
        if not item_field:
            if self.IsList():
                log.debug('Unable to flatten list response, raw results being returned.')
            return self._NormalRequest(service, request)
        if not self.HasTokenizedRequest():
            if self.IsList():
                return self._FlatNonPagedRequest(service, request, item_field)
            else:
                return self._NormalRequest(service, request)

        def RequestFunc(global_params=None):
            return list_pager.YieldFromList(service, request, method=self._method_name, field=item_field, global_params=global_params, limit=limit, current_token_attribute='pageToken', next_token_attribute='nextPageToken', batch_size_attribute=self.BatchPageSizeField(), batch_size=page_size)
        return RequestFunc

    def _NormalRequest(self, service, request):
        """Generates a basic request function for the method.

    Args:
      service: The apitools service that will be making the request.
      request: The apitools request object to send.

    Returns:
      A function to make the request.
    """

        def RequestFunc(global_params=None):
            method = getattr(service, self._method_name)
            return method(request, global_params=global_params)
        return RequestFunc

    def _FlatNonPagedRequest(self, service, request, item_field):
        """Generates a request function for the method that extracts an item list.

    List responses usually have a single repeated field that represents the
    actual items being listed. This request function returns only those items
    not the entire response.

    Args:
      service: The apitools service that will be making the request.
      request: The apitools request object to send.
      item_field: str, The name of the field that the list of items can be found
       in.

    Returns:
      A function to make the request.
    """

        def RequestFunc(global_params=None):
            response = self._NormalRequest(service, request)(global_params=global_params)
            return getattr(response, item_field)
        return RequestFunc