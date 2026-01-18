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
class HostedFile(Resource):
    """A resource representing a file managed by a lazr.restful service."""

    def open(self, mode='r', content_type=None, filename=None):
        """Open the file on the server for read or write access."""
        if mode in ('r', 'w'):
            return HostedFileBuffer(self, mode, content_type, filename)
        else:
            raise ValueError('Invalid mode. Supported modes are: r, w')

    def delete(self):
        """Delete the file from the server."""
        self._root._browser.delete(self._wadl_resource.url)

    def _get_parameter_names(self, *kinds):
        """HostedFile objects define no web service parameters."""
        return []

    def __eq__(self, other):
        """Equality comparison.

        Two hosted files are the same if they have the same URL.

        There is no need to check the contents because the only way to
        retrieve or modify the hosted file contents is to open a
        filehandle, which goes direct to the server.
        """
        return other is not None and self._wadl_resource.url == other._wadl_resource.url