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
class RequestMockBuilder(object):
    """A simple mock of HttpRequest

    Pass in a dictionary to the constructor that maps request methodIds to
    tuples of (httplib2.Response, content, opt_expected_body) that should be
    returned when that method is called. None may also be passed in for the
    httplib2.Response, in which case a 200 OK response will be generated.
    If an opt_expected_body (str or dict) is provided, it will be compared to
    the body and UnexpectedBodyError will be raised on inequality.

    Example:
      response = '{"data": {"id": "tag:google.c...'
      requestBuilder = RequestMockBuilder(
        {
          'plus.activities.get': (None, response),
        }
      )
      googleapiclient.discovery.build("plus", "v1", requestBuilder=requestBuilder)

    Methods that you do not supply a response for will return a
    200 OK with an empty string as the response content or raise an excpetion
    if check_unexpected is set to True. The methodId is taken from the rpcName
    in the discovery document.

    For more details see the project wiki.
  """

    def __init__(self, responses, check_unexpected=False):
        """Constructor for RequestMockBuilder

    The constructed object should be a callable object
    that can replace the class HttpResponse.

    responses - A dictionary that maps methodIds into tuples
                of (httplib2.Response, content). The methodId
                comes from the 'rpcName' field in the discovery
                document.
    check_unexpected - A boolean setting whether or not UnexpectedMethodError
                       should be raised on unsupplied method.
    """
        self.responses = responses
        self.check_unexpected = check_unexpected

    def __call__(self, http, postproc, uri, method='GET', body=None, headers=None, methodId=None, resumable=None):
        """Implements the callable interface that discovery.build() expects
    of requestBuilder, which is to build an object compatible with
    HttpRequest.execute(). See that method for the description of the
    parameters and the expected response.
    """
        if methodId in self.responses:
            response = self.responses[methodId]
            resp, content = response[:2]
            if len(response) > 2:
                expected_body = response[2]
                if bool(expected_body) != bool(body):
                    raise UnexpectedBodyError(expected_body, body)
                if isinstance(expected_body, str):
                    expected_body = json.loads(expected_body)
                body = json.loads(body)
                if body != expected_body:
                    raise UnexpectedBodyError(expected_body, body)
            return HttpRequestMock(resp, content, postproc)
        elif self.check_unexpected:
            raise UnexpectedMethodError(methodId=methodId)
        else:
            model = JsonModel(False)
            return HttpRequestMock(None, '{}', model.response)