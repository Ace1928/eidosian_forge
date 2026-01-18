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
def CheckResponse(response):
    if response is None:
        raise exceptions.RequestError('Request to url %s did not return a response.' % response.request_url)