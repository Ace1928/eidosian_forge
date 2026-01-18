from __future__ import absolute_import
import asyncio
import functools
import aiohttp  # type: ignore
import six
import urllib3  # type: ignore
from google.auth import exceptions
from google.auth import transport
from google.auth.transport import requests
class _CombinedResponse(transport.Response):
    """
    In order to more closely resemble the `requests` interface, where a raw
    and deflated content could be accessed at once, this class lazily reads the
    stream in `transport.Response` so both return forms can be used.

    The gzip and deflate transfer-encodings are automatically decoded for you
    because the default parameter for autodecompress into the ClientSession is set
    to False, and therefore we add this class to act as a wrapper for a user to be
    able to access both the raw and decoded response bodies - mirroring the sync
    implementation.
    """

    def __init__(self, response):
        self._response = response
        self._raw_content = None

    def _is_compressed(self):
        headers = self._response.headers
        return 'Content-Encoding' in headers and (headers['Content-Encoding'] == 'gzip' or headers['Content-Encoding'] == 'deflate')

    @property
    def status(self):
        return self._response.status

    @property
    def headers(self):
        return self._response.headers

    @property
    def data(self):
        return self._response.content

    async def raw_content(self):
        if self._raw_content is None:
            self._raw_content = await self._response.content.read()
        return self._raw_content

    async def content(self):
        await self.raw_content()
        if self._is_compressed():
            decoder = urllib3.response.MultiDecoder(self._response.headers['Content-Encoding'])
            decompressed = decoder.decompress(self._raw_content)
            return decompressed
        return self._raw_content