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
class MediaFileUpload(MediaIoBaseUpload):
    """A MediaUpload for a file.

  Construct a MediaFileUpload and pass as the media_body parameter of the
  method. For example, if we had a service that allowed uploading images:

    media = MediaFileUpload('cow.png', mimetype='image/png',
      chunksize=1024*1024, resumable=True)
    farm.animals().insert(
        id='cow',
        name='cow.png',
        media_body=media).execute()

  Depending on the platform you are working on, you may pass -1 as the
  chunksize, which indicates that the entire file should be uploaded in a single
  request. If the underlying platform supports streams, such as Python 2.6 or
  later, then this can be very efficient as it avoids multiple connections, and
  also avoids loading the entire file into memory before sending it. Note that
  Google App Engine has a 5MB limit on request size, so you should never set
  your chunksize larger than 5MB, or to -1.
  """

    @util.positional(2)
    def __init__(self, filename, mimetype=None, chunksize=DEFAULT_CHUNK_SIZE, resumable=False):
        """Constructor.

    Args:
      filename: string, Name of the file.
      mimetype: string, Mime-type of the file. If None then a mime-type will be
        guessed from the file extension.
      chunksize: int, File will be uploaded in chunks of this many bytes. Only
        used if resumable=True. Pass in a value of -1 if the file is to be
        uploaded in a single chunk. Note that Google App Engine has a 5MB limit
        on request size, so you should never set your chunksize larger than 5MB,
        or to -1.
      resumable: bool, True if this is a resumable upload. False means upload
        in a single request.
    """
        self._fd = None
        self._filename = filename
        self._fd = open(self._filename, 'rb')
        if mimetype is None:
            mimetype, _ = mimetypes.guess_type(filename)
            if mimetype is None:
                mimetype = 'application/octet-stream'
        super(MediaFileUpload, self).__init__(self._fd, mimetype, chunksize=chunksize, resumable=resumable)

    def __del__(self):
        if self._fd:
            self._fd.close()

    def to_json(self):
        """Creating a JSON representation of an instance of MediaFileUpload.

    Returns:
       string, a JSON representation of this instance, suitable to pass to
       from_json().
    """
        return self._to_json(strip=['_fd'])

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        return MediaFileUpload(d['_filename'], mimetype=d['_mimetype'], chunksize=d['_chunksize'], resumable=d['_resumable'])