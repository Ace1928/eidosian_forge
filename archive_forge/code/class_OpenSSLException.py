from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import subprocess
import tempfile
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import log
import six
class OpenSSLException(exceptions.Error):
    """Exception for problems with OpenSSL functions."""