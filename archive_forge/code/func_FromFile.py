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
def FromFile(cls, filename, mime_type=None, auto_transfer=True, gzip_encoded=False, **kwds):
    """Create a new Upload object from a filename."""
    path = os.path.expanduser(filename)
    if not os.path.exists(path):
        raise exceptions.NotFoundError('Could not find file %s' % path)
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            raise exceptions.InvalidUserInputError('Could not guess mime type for %s' % path)
    size = os.stat(path).st_size
    return cls(open(path, 'rb'), mime_type, total_size=size, close_stream=True, auto_transfer=auto_transfer, gzip_encoded=gzip_encoded, **kwds)