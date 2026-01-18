from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import struct
import sys
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
import six
from six.moves.urllib import parse
import socks
def CheckPythonVersion(ignore_certs):
    if not ignore_certs and (six.PY2 and sys.version_info < (2, 7, 9) or (six.PY3 and sys.version_info < (3, 2, 0))):
        raise PythonVersionMissingSNI('Python version %d.%d.%d does not support SSL/TLS SNI needed for certificate verification on WebSocket connection.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro))