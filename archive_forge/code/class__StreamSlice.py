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
class _StreamSlice(object):
    """Truncated stream.

  Takes a stream and presents a stream that is a slice of the original stream.
  This is used when uploading media in chunks. In later versions of Python a
  stream can be passed to httplib in place of the string of data to send. The
  problem is that httplib just blindly reads to the end of the stream. This
  wrapper presents a virtual stream that only reads to the end of the chunk.
  """

    def __init__(self, stream, begin, chunksize):
        """Constructor.

    Args:
      stream: (io.Base, file object), the stream to wrap.
      begin: int, the seek position the chunk begins at.
      chunksize: int, the size of the chunk.
    """
        self._stream = stream
        self._begin = begin
        self._chunksize = chunksize
        self._stream.seek(begin)

    def read(self, n=-1):
        """Read n bytes.

    Args:
      n, int, the number of bytes to read.

    Returns:
      A string of length 'n', or less if EOF is reached.
    """
        cur = self._stream.tell()
        end = self._begin + self._chunksize
        if n == -1 or cur + n > end:
            n = end - cur
        return self._stream.read(n)