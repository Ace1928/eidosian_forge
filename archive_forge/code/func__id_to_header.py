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
def _id_to_header(self, id_):
    """Convert an id to a Content-ID header value.

    Args:
      id_: string, identifier of individual request.

    Returns:
      A Content-ID header with the id_ encoded into it. A UUID is prepended to
      the value because Content-ID headers are supposed to be universally
      unique.
    """
    if self._base_id is None:
        self._base_id = uuid.uuid4()
    return '<%s + %s>' % (self._base_id, quote(id_))