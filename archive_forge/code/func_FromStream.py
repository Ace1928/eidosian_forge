from __future__ import print_function
import email.generator as email_generator
import email.mime.multipart as mime_multipart
import email.mime.nonmultipart as mime_nonmultipart
import io
import json
import mimetypes
import os
import threading
import six
from six.moves import http_client
from apitools.base.py import buffered_stream
from apitools.base.py import compression
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import stream_slice
from apitools.base.py import util
@classmethod
def FromStream(cls, stream, mime_type, total_size=None, auto_transfer=True, gzip_encoded=False, **kwds):
    """Create a new Upload object from a stream."""
    if mime_type is None:
        raise exceptions.InvalidUserInputError('No mime_type specified for stream')
    return cls(stream, mime_type, total_size=total_size, close_stream=False, auto_transfer=auto_transfer, gzip_encoded=gzip_encoded, **kwds)