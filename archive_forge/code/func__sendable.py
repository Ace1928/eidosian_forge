import collections.abc
import copy
import errno
import functools
import http.client
import os
import re
import urllib.parse as urlparse
import osprofiler.web
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import netutils
from glance.common import auth
from glance.common import exception
from glance.common import utils
from glance.i18n import _
def _sendable(self, body):
    return SENDFILE_SUPPORTED and hasattr(body, 'fileno') and self._seekable(body) and (not self.use_ssl)