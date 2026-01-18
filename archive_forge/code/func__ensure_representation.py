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
def _ensure_representation(self):
    """Make sure this resource has a representation fetched."""
    if self._wadl_resource.representation is None:
        representation = self._root._browser.get(self._wadl_resource)
        if isinstance(representation, binary_type):
            representation = representation.decode('utf-8')
        representation = loads(representation)
        if isinstance(representation, dict):
            type_link = representation['resource_type_link']
            if type_link is not None and type_link != self._wadl_resource.type_url:
                resource_type = self._root._wadl.get_resource_type(type_link)
                self._wadl_resource.tag = resource_type.tag
        self.__dict__['_wadl_resource'] = self._wadl_resource.bind(representation, self.JSON_MEDIA_TYPE, representation_needs_processing=False)