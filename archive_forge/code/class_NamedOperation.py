from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
class NamedOperation(RestfulBase):
    """A class for a named operation to be invoked with GET or POST."""

    def __init__(self, root, resource, wadl_method):
        """Initialize with respect to a WADL Method object"""
        self.root = root
        self.resource = resource
        self.wadl_method = wadl_method

    def __call__(self, *args, **kwargs):
        """Invoke the method and process the result."""
        if len(args) > 0:
            raise TypeError('Method must be called with keyword args.')
        http_method = self.wadl_method.name
        args = self._transform_resources_to_links(kwargs)
        request = self.wadl_method.request
        if http_method in ('get', 'head', 'delete'):
            params = request.query_params
        else:
            definition = request.get_representation_definition('multipart/form-data')
            if definition is None:
                definition = request.get_representation_definition('application/x-www-form-urlencoded')
            assert definition is not None, 'A POST named operation must define a multipart or form-urlencoded request representation.'
            params = definition.params(self.resource._wadl_resource)
        send_as_is_params = {param.name for param in params if param.type == 'binary' or len(param.options) > 0}
        for key, value in args.items():
            if key not in send_as_is_params:
                args[key] = dumps(value, cls=DatetimeJSONEncoder)
        if http_method in ('get', 'head', 'delete'):
            url = self.wadl_method.build_request_url(**args)
            in_representation = ''
            extra_headers = {}
        else:
            url = self.wadl_method.build_request_url()
            media_type, in_representation = self.wadl_method.build_representation(**args)
            extra_headers = {'Content-type': media_type}
        response, content = self.root._browser._request(url, in_representation, http_method.upper(), extra_headers=extra_headers)
        if response.status == 201:
            return self._handle_201_response(url, response, content)
        else:
            if http_method == 'post':
                if response.status == 301:
                    url = response['location']
                    response, content = self.root._browser._request(url)
                else:
                    self.resource.lp_refresh()
            return self._handle_200_response(url, response, content)

    def _handle_201_response(self, url, response, content):
        """Handle the creation of a new resource by fetching it."""
        wadl_response = self.wadl_method.response.bind(HeaderDictionary(response))
        wadl_parameter = wadl_response.get_parameter('Location')
        wadl_resource = wadl_parameter.linked_resource
        response, content = self.root._browser._request(wadl_resource.url)
        return Resource._create_bound_resource(self.root, wadl_resource, content, response['content-type'])

    def _handle_200_response(self, url, response, content):
        """Process the return value of an operation."""
        content_type = response['content-type']
        response_definition = self.wadl_method.response
        representation_definition = response_definition.get_representation_definition(content_type)
        if representation_definition is None:
            if content_type == self.JSON_MEDIA_TYPE:
                if isinstance(content, binary_type):
                    content = content.decode('utf-8')
                return loads(content)
            return content
        if isinstance(content, binary_type):
            content = content.decode('utf-8')
        document = loads(content)
        if document is None:
            return document
        if 'self_link' in document and 'resource_type_link' in document:
            url = document['self_link']
            resource_type = self.root._wadl.get_resource_type(document['resource_type_link'])
            wadl_resource = WadlResource(self.root._wadl, url, resource_type.tag)
        else:
            representation_definition = representation_definition.resolve_definition()
            wadl_resource = WadlResource(self.root._wadl, url, representation_definition.tag)
        return Resource._create_bound_resource(self.root, wadl_resource, document, content_type, representation_needs_processing=False, representation_definition=representation_definition)

    def _get_external_param_name(self, param_name):
        """Named operation parameter names are sent as is."""
        return param_name