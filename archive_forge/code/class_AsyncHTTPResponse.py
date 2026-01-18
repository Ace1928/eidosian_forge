from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
class AsyncHTTPResponse:
    """Async HTTP Response."""

    def __init__(self, response):
        self.response = response
        self._msg = None
        self.version = 10

    def read(self, *args, **kwargs):
        return self.response.body

    def getheader(self, name, default=None):
        return self.response.headers.get(name, default)

    def getheaders(self):
        return list(self.response.headers.items())

    @property
    def msg(self):
        if self._msg is None:
            self._msg = MIMEMessage(message_from_headers(self.getheaders()))
        return self._msg

    @property
    def status(self):
        return self.response.code

    @property
    def reason(self):
        if self.response.error:
            return self.response.error.message
        return ''

    def __repr__(self):
        return repr(self.response)