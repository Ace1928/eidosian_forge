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