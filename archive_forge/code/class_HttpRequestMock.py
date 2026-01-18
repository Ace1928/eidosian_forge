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
class HttpRequestMock(object):
    """Mock of HttpRequest.

  Do not construct directly, instead use RequestMockBuilder.
  """

    def __init__(self, resp, content, postproc):
        """Constructor for HttpRequestMock

    Args:
      resp: httplib2.Response, the response to emulate coming from the request
      content: string, the response body
      postproc: callable, the post processing function usually supplied by
                the model class. See model.JsonModel.response() as an example.
    """
        self.resp = resp
        self.content = content
        self.postproc = postproc
        if resp is None:
            self.resp = httplib2.Response({'status': 200, 'reason': 'OK'})
        if 'reason' in self.resp:
            self.resp.reason = self.resp['reason']

    def execute(self, http=None):
        """Execute the request.

    Same behavior as HttpRequest.execute(), but the response is
    mocked and not really from an HTTP request/response.
    """
        return self.postproc(self.resp, self.content)