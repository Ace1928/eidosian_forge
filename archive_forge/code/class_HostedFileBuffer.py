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
class HostedFileBuffer(BytesIO):
    """The contents of a file hosted by a lazr.restful service."""

    def __init__(self, hosted_file, mode, content_type=None, filename=None):
        self.url = hosted_file._wadl_resource.url
        if mode == 'r':
            if content_type is not None:
                raise ValueError("Files opened for read access can't specify content_type.")
            if filename is not None:
                raise ValueError("Files opened for read access can't specify filename.")
            response, value = hosted_file._root._browser.get(self.url, return_response=True)
            content_type = response['content-type']
            last_modified = response.get('last-modified')
            content_location = response['content-location']
            path = urlparse(content_location)[2]
            filename = unquote(path.split('/')[-1])
        elif mode == 'w':
            value = b''
            if content_type is None:
                raise ValueError('Files opened for write access must specify content_type.')
            if filename is None:
                raise ValueError('Files opened for write access must specify filename.')
            last_modified = None
        else:
            raise ValueError('Invalid mode. Supported modes are: r, w')
        self.hosted_file = hosted_file
        self.mode = mode
        self.content_type = content_type
        self.filename = filename
        self.last_modified = last_modified
        BytesIO.__init__(self, value)

    def close(self):
        if self.mode == 'w':
            disposition = 'attachment; filename="%s"' % self.filename
            self.hosted_file._root._browser.put(self.url, self.getvalue(), self.content_type, {'Content-Disposition': disposition})
        BytesIO.close(self)

    def write(self, b):
        BytesIO.write(self, b)