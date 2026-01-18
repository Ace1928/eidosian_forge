import errno
from eventlet.green import socket
import functools
import os
import re
import urllib
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import netutils
from oslo_utils import strutils
from webob import exc
from glance.common import exception
from glance.common import location_strategy
from glance.common import timeutils
from glance.common import wsgi
from glance.i18n import _, _LE, _LW
class LimitingReader(object):
    """
    Reader designed to fail when reading image data past the configured
    allowable amount.
    """

    def __init__(self, data, limit, exception_class=exception.ImageSizeLimitExceeded):
        """
        :param data: Underlying image data object
        :param limit: maximum number of bytes the reader should allow
        :param exception_class: Type of exception to be raised
        """
        self.data = data
        self.limit = limit
        self.bytes_read = 0
        self.exception_class = exception_class

    def __iter__(self):
        for chunk in self.data:
            self.bytes_read += len(chunk)
            if self.bytes_read > self.limit:
                raise self.exception_class()
            else:
                yield chunk

    def read(self, i):
        result = self.data.read(i)
        self.bytes_read += len(result)
        if self.bytes_read > self.limit:
            raise self.exception_class()
        return result