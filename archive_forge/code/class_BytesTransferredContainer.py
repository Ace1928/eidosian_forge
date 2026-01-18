from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import copy
import logging
import re
import socket
import types
import six
from six.moves import http_client
from six.moves import urllib
from six.moves import cStringIO
from apitools.base.py import exceptions as apitools_exceptions
from gslib.cloud_api import BadRequestException
from gslib.lazy_wrapper import LazyWrapper
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.utils.constants import DEBUGLEVEL_DUMP_REQUESTS
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.constants import UTF8
from gslib.utils import text_util
import httplib2
from httplib2 import parse_uri
class BytesTransferredContainer(object):
    """Container class for passing number of bytes transferred to lower layers.

  For resumed transfers or connection rebuilds in the middle of a transfer, we
  need to rebuild the connection class with how much we've transferred so far.
  For uploads, we don't know the total number of bytes uploaded until we've
  queried the server, but we need to create the connection class to pass to
  httplib2 before we can query the server. This container object allows us to
  pass a reference into Upload/DownloadCallbackConnection.
  """

    def __init__(self):
        self.__bytes_transferred = 0

    @property
    def bytes_transferred(self):
        return self.__bytes_transferred

    @bytes_transferred.setter
    def bytes_transferred(self, value):
        self.__bytes_transferred = value