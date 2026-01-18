import re
from io import BytesIO
from urllib.parse import unquote
from twisted.internet.protocol import ClientCreator, Protocol
from twisted.protocols.ftp import CommandFailed, FTPClient
from scrapy.http import Response
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes
class ReceivedDataProtocol(Protocol):

    def __init__(self, filename=None):
        self.__filename = filename
        self.body = open(filename, 'wb') if filename else BytesIO()
        self.size = 0

    def dataReceived(self, data):
        self.body.write(data)
        self.size += len(data)

    @property
    def filename(self):
        return self.__filename

    def close(self):
        self.body.close() if self.filename else self.body.seek(0)