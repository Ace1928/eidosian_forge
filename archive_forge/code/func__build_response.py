import re
from io import BytesIO
from urllib.parse import unquote
from twisted.internet.protocol import ClientCreator, Protocol
from twisted.protocols.ftp import CommandFailed, FTPClient
from scrapy.http import Response
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes
def _build_response(self, result, request, protocol):
    self.result = result
    protocol.close()
    headers = {'local filename': protocol.filename or '', 'size': protocol.size}
    body = to_bytes(protocol.filename or protocol.body.read())
    respcls = responsetypes.from_args(url=request.url, body=body)
    return respcls(url=request.url, status=200, body=body, headers=headers)