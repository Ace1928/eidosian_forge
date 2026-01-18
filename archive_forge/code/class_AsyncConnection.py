from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
class AsyncConnection:
    """Async AWS Connection."""

    def __init__(self, sqs_connection, http_client=None, **kwargs):
        self.sqs_connection = sqs_connection
        self._httpclient = http_client or get_client()

    def get_http_connection(self):
        return AsyncHTTPSConnection(http_client=self._httpclient)

    def _mexe(self, request, sender=None, callback=None):
        callback = callback or promise()
        conn = self.get_http_connection()
        if callable(sender):
            sender(conn, request.method, request.path, request.body, request.headers, callback)
        else:
            conn.request(request.method, request.url, request.body, request.headers)
            conn.getresponse(callback=callback)
        return callback