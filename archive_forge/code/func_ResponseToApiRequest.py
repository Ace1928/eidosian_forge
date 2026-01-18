from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import requests
from six.moves import urllib
def ResponseToApiRequest(identifiers, entity_path, entity_collection=None, method='GET', query_params=None, accept_mimetype=None, body=None, body_mimetype='application/json'):
    """Makes a request to the Apigee API and returns the response.

  Args:
    identifiers: a collection that maps entity type names to identifiers.
    entity_path: a list of entity type names from least to most specific.
    entity_collection: if provided, the final entity type; the request will not
      be specific as to which entity of that type is being referenced.
    method: an HTTP method string specifying what to do with the accessed
      entity. If the method begins with a colon, it will be interpreted as a
      Cloud custom method (https://cloud.google.com/apis/design/custom_methods)
      and appended to the request URL with the POST HTTP method.
    query_params: any extra query parameters to be sent in the request.
    accept_mimetype: the mimetype to expect in the response body. If not
      provided, the response will be parsed as JSON.
    body: data to send in the request body.
    body_mimetype: the mimetype of the body data, if not JSON.

  Returns:
    an object containing the API's response. If accept_mimetype was set, this
      will be raw bytes. Otherwise, it will be a parsed JSON object.

  Raises:
    MissingIdentifierError: an entry in entity_path is missing from
      `identifiers`.
    RequestError: if the request itself fails.
  """
    headers = {}
    if body:
        headers['Content-Type'] = body_mimetype
    if accept_mimetype:
        headers['Accept'] = accept_mimetype
    resource_identifier = _ResourceIdentifier(identifiers, entity_path)
    url_path_elements = ['v1']
    for key, value in resource_identifier.items():
        url_path_elements += [key.plural, urllib.parse.quote(value)]
    if entity_collection:
        collection_name = resource_args.ENTITIES[entity_collection].plural
        url_path_elements.append(urllib.parse.quote(collection_name))
    query_string = urllib.parse.urlencode(query_params) if query_params else ''
    endpoint_override = properties.VALUES.api_endpoint_overrides.apigee.Get()
    if endpoint_override:
        endpoint = urllib.parse.urlparse(endpoint_override)
        scheme = endpoint.scheme
        host = endpoint.netloc
    else:
        scheme = 'https'
        host = APIGEE_HOST
    url_path = '/'.join(url_path_elements)
    if method and method[0] == ':':
        url_path += method
        method = 'POST'
    url = urllib.parse.urlunparse((scheme, host, url_path, '', query_string, ''))
    status, reason, response = _Communicate(url, method, body, headers)
    if status >= 400:
        resource_type = _GetResourceType(entity_collection, entity_path)
        if status == 404:
            exception_class = errors.EntityNotFoundError
        elif status in (401, 403):
            exception_class = errors.UnauthorizedRequestError
        else:
            exception_class = errors.RequestError
        error_identifier = _BuildErrorIdentifier(resource_identifier)
        try:
            user_help = _ExtractErrorMessage(_DecodeResponse(response))
        except ValueError:
            user_help = None
        raise exception_class(resource_type, error_identifier, method, reason, response, user_help=user_help)
    if accept_mimetype is None:
        try:
            response = _DecodeResponse(response)
            response = json.loads(response)
        except ValueError as error:
            resource_type = _GetResourceType(entity_collection, entity_path)
            error_identifier = _BuildErrorIdentifier(resource_identifier)
            raise errors.ResponseNotJSONError(error, resource_type, error_identifier, response)
    return response