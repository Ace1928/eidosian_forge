from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import platform
import re
import time
import uuid
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
def AppendToHeader(header, value):
    """Appends the given value to the existing value in the http request.

  Args:
    header: str, The name of the header to append to.
    value: str, The value to append to the existing header value.

  Returns:
    A function that can be used in a Handler.request.
  """
    header, value = _EncodeHeader(header, value)

    def _AppendToHeader(request):
        """Appends a value to a header on a request."""
        headers = request.headers
        current_value = b''
        for hdr, v in six.iteritems(headers):
            if hdr.lower() == header.lower():
                current_value = v
                del headers[hdr]
                break
        headers[header] = (current_value + b' ' + value).strip() if current_value else value
    return _AppendToHeader