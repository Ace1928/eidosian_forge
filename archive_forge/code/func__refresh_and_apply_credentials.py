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
def _refresh_and_apply_credentials(self, request, http):
    """Refresh the credentials and apply to the request.

    Args:
      request: HttpRequest, the request.
      http: httplib2.Http, the global http object for the batch.
    """
    creds = None
    request_credentials = False
    if request.http is not None:
        creds = _auth.get_credentials_from_http(request.http)
        request_credentials = True
    if creds is None and http is not None:
        creds = _auth.get_credentials_from_http(http)
    if creds is not None:
        if id(creds) not in self._refreshed_credentials:
            _auth.refresh_credentials(creds)
            self._refreshed_credentials[id(creds)] = 1
    if request.http is None or not request_credentials:
        _auth.apply_credentials(creds, request.headers)