from __future__ import absolute_import
import six
from six.moves import http_client
from six.moves import range
from six import BytesIO, StringIO
from six.moves.urllib.parse import urlparse, urlunparse, quote, unquote
import copy
import httplib2
import json
import logging
import mimetypes
import os
import random
import socket
import time
import uuid
from email.generator import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.parser import FeedParser
from googleapiclient import _helpers as util
from googleapiclient import _auth
from googleapiclient.errors import BatchError
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidChunkSizeError
from googleapiclient.errors import ResumableUploadError
from googleapiclient.errors import UnexpectedBodyError
from googleapiclient.errors import UnexpectedMethodError
from googleapiclient.model import JsonModel
class HttpMockSequence(object):
    """Mock of httplib2.Http

  Mocks a sequence of calls to request returning different responses for each
  call. Create an instance initialized with the desired response headers
  and content and then use as if an httplib2.Http instance.

    http = HttpMockSequence([
      ({'status': '401'}, ''),
      ({'status': '200'}, '{"access_token":"1/3w","expires_in":3600}'),
      ({'status': '200'}, 'echo_request_headers'),
      ])
    resp, content = http.request("http://examples.com")

  There are special values you can pass in for content to trigger
  behavours that are helpful in testing.

  'echo_request_headers' means return the request headers in the response body
  'echo_request_headers_as_json' means return the request headers in
     the response body
  'echo_request_body' means return the request body in the response body
  'echo_request_uri' means return the request uri in the response body
  """

    def __init__(self, iterable):
        """
    Args:
      iterable: iterable, a sequence of pairs of (headers, body)
    """
        self._iterable = iterable
        self.follow_redirects = True
        self.request_sequence = list()

    def request(self, uri, method='GET', body=None, headers=None, redirections=1, connection_type=None):
        self.request_sequence.append((uri, method, body, headers))
        resp, content = self._iterable.pop(0)
        content = six.ensure_binary(content)
        if content == b'echo_request_headers':
            content = headers
        elif content == b'echo_request_headers_as_json':
            content = json.dumps(headers)
        elif content == b'echo_request_body':
            if hasattr(body, 'read'):
                content = body.read()
            else:
                content = body
        elif content == b'echo_request_uri':
            content = uri
        if isinstance(content, six.text_type):
            content = content.encode('utf-8')
        return (httplib2.Response(resp), content)

    def close(self):
        return None