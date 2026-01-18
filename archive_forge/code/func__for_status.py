from __future__ import annotations
from email import message_from_bytes
from email.mime.message import MIMEMessage
from vine import promise, transform
from kombu.asynchronous.aws.ext import AWSRequest, get_response
from kombu.asynchronous.http import Headers, Request, get_client
def _for_status(self, response, body):
    context = 'Empty body' if not body else 'HTTP Error'
    return Exception('Request {}  HTTP {}  {} ({})'.format(context, response.status, response.reason, body))