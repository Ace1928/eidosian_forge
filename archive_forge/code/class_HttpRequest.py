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
class HttpRequest(object):
    """Encapsulates a single HTTP request."""

    @util.positional(4)
    def __init__(self, http, postproc, uri, method='GET', body=None, headers=None, methodId=None, resumable=None):
        """Constructor for an HttpRequest.

    Args:
      http: httplib2.Http, the transport object to use to make a request
      postproc: callable, called on the HTTP response and content to transform
                it into a data object before returning, or raising an exception
                on an error.
      uri: string, the absolute URI to send the request to
      method: string, the HTTP method to use
      body: string, the request body of the HTTP request,
      headers: dict, the HTTP request headers
      methodId: string, a unique identifier for the API method being called.
      resumable: MediaUpload, None if this is not a resumbale request.
    """
        self.uri = uri
        self.method = method
        self.body = body
        self.headers = headers or {}
        self.methodId = methodId
        self.http = http
        self.postproc = postproc
        self.resumable = resumable
        self.response_callbacks = []
        self._in_error_state = False
        self.body_size = len(self.body or '')
        self.resumable_uri = None
        self.resumable_progress = 0
        self._rand = random.random
        self._sleep = time.sleep

    @util.positional(1)
    def execute(self, http=None, num_retries=0):
        """Execute the request.

    Args:
      http: httplib2.Http, an http object to be used in place of the
            one the HttpRequest request object was constructed with.
      num_retries: Integer, number of times to retry with randomized
            exponential backoff. If all retries fail, the raised HttpError
            represents the last request. If zero (default), we attempt the
            request only once.

    Returns:
      A deserialized object model of the response body as determined
      by the postproc.

    Raises:
      googleapiclient.errors.HttpError if the response was not a 2xx.
      httplib2.HttpLib2Error if a transport error has occurred.
    """
        if http is None:
            http = self.http
        if self.resumable:
            body = None
            while body is None:
                _, body = self.next_chunk(http=http, num_retries=num_retries)
            return body
        if 'content-length' not in self.headers:
            self.headers['content-length'] = str(self.body_size)
        if len(self.uri) > MAX_URI_LENGTH and self.method == 'GET':
            self.method = 'POST'
            self.headers['x-http-method-override'] = 'GET'
            self.headers['content-type'] = 'application/x-www-form-urlencoded'
            parsed = urlparse(self.uri)
            self.uri = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, None, None))
            self.body = parsed.query
            self.headers['content-length'] = str(len(self.body))
        resp, content = _retry_request(http, num_retries, 'request', self._sleep, self._rand, str(self.uri), method=str(self.method), body=self.body, headers=self.headers)
        for callback in self.response_callbacks:
            callback(resp)
        if resp.status >= 300:
            raise HttpError(resp, content, uri=self.uri)
        return self.postproc(resp, content)

    @util.positional(2)
    def add_response_callback(self, cb):
        """add_response_headers_callback

    Args:
      cb: Callback to be called on receiving the response headers, of signature:

      def cb(resp):
        # Where resp is an instance of httplib2.Response
    """
        self.response_callbacks.append(cb)

    @util.positional(1)
    def next_chunk(self, http=None, num_retries=0):
        """Execute the next step of a resumable upload.

    Can only be used if the method being executed supports media uploads and
    the MediaUpload object passed in was flagged as using resumable upload.

    Example:

      media = MediaFileUpload('cow.png', mimetype='image/png',
                              chunksize=1000, resumable=True)
      request = farm.animals().insert(
          id='cow',
          name='cow.png',
          media_body=media)

      response = None
      while response is None:
        status, response = request.next_chunk()
        if status:
          print "Upload %d%% complete." % int(status.progress() * 100)


    Args:
      http: httplib2.Http, an http object to be used in place of the
            one the HttpRequest request object was constructed with.
      num_retries: Integer, number of times to retry with randomized
            exponential backoff. If all retries fail, the raised HttpError
            represents the last request. If zero (default), we attempt the
            request only once.

    Returns:
      (status, body): (ResumableMediaStatus, object)
         The body will be None until the resumable media is fully uploaded.

    Raises:
      googleapiclient.errors.HttpError if the response was not a 2xx.
      httplib2.HttpLib2Error if a transport error has occurred.
    """
        if http is None:
            http = self.http
        if self.resumable.size() is None:
            size = '*'
        else:
            size = str(self.resumable.size())
        if self.resumable_uri is None:
            start_headers = copy.copy(self.headers)
            start_headers['X-Upload-Content-Type'] = self.resumable.mimetype()
            if size != '*':
                start_headers['X-Upload-Content-Length'] = size
            start_headers['content-length'] = str(self.body_size)
            resp, content = _retry_request(http, num_retries, 'resumable URI request', self._sleep, self._rand, self.uri, method=self.method, body=self.body, headers=start_headers)
            if resp.status == 200 and 'location' in resp:
                self.resumable_uri = resp['location']
            else:
                raise ResumableUploadError(resp, content)
        elif self._in_error_state:
            headers = {'Content-Range': 'bytes */%s' % size, 'content-length': '0'}
            resp, content = http.request(self.resumable_uri, 'PUT', headers=headers)
            status, body = self._process_response(resp, content)
            if body:
                return (status, body)
        if self.resumable.has_stream():
            data = self.resumable.stream()
            if self.resumable.chunksize() == -1:
                data.seek(self.resumable_progress)
                chunk_end = self.resumable.size() - self.resumable_progress - 1
            else:
                data = _StreamSlice(data, self.resumable_progress, self.resumable.chunksize())
                chunk_end = min(self.resumable_progress + self.resumable.chunksize() - 1, self.resumable.size() - 1)
        else:
            data = self.resumable.getbytes(self.resumable_progress, self.resumable.chunksize())
            if len(data) < self.resumable.chunksize():
                size = str(self.resumable_progress + len(data))
            chunk_end = self.resumable_progress + len(data) - 1
        headers = {'Content-Length': str(chunk_end - self.resumable_progress + 1)}
        if chunk_end != -1:
            headers['Content-Range'] = 'bytes %d-%d/%s' % (self.resumable_progress, chunk_end, size)
        for retry_num in range(num_retries + 1):
            if retry_num > 0:
                self._sleep(self._rand() * 2 ** retry_num)
                LOGGER.warning('Retry #%d for media upload: %s %s, following status: %d' % (retry_num, self.method, self.uri, resp.status))
            try:
                resp, content = http.request(self.resumable_uri, method='PUT', body=data, headers=headers)
            except:
                self._in_error_state = True
                raise
            if not _should_retry_response(resp.status, content):
                break
        return self._process_response(resp, content)

    def _process_response(self, resp, content):
        """Process the response from a single chunk upload.

    Args:
      resp: httplib2.Response, the response object.
      content: string, the content of the response.

    Returns:
      (status, body): (ResumableMediaStatus, object)
         The body will be None until the resumable media is fully uploaded.

    Raises:
      googleapiclient.errors.HttpError if the response was not a 2xx or a 308.
    """
        if resp.status in [200, 201]:
            self._in_error_state = False
            return (None, self.postproc(resp, content))
        elif resp.status == 308:
            self._in_error_state = False
            try:
                self.resumable_progress = int(resp['range'].split('-')[1]) + 1
            except KeyError:
                self.resumable_progress = 0
            if 'location' in resp:
                self.resumable_uri = resp['location']
        else:
            self._in_error_state = True
            raise HttpError(resp, content, uri=self.uri)
        return (MediaUploadProgress(self.resumable_progress, self.resumable.size()), None)

    def to_json(self):
        """Returns a JSON representation of the HttpRequest."""
        d = copy.copy(self.__dict__)
        if d['resumable'] is not None:
            d['resumable'] = self.resumable.to_json()
        del d['http']
        del d['postproc']
        del d['_sleep']
        del d['_rand']
        return json.dumps(d)

    @staticmethod
    def from_json(s, http, postproc):
        """Returns an HttpRequest populated with info from a JSON object."""
        d = json.loads(s)
        if d['resumable'] is not None:
            d['resumable'] = MediaUpload.new_from_json(d['resumable'])
        return HttpRequest(http, postproc, uri=d['uri'], method=d['method'], body=d['body'], headers=d['headers'], methodId=d['methodId'], resumable=d['resumable'])

    @staticmethod
    def null_postproc(resp, contents):
        return (resp, contents)