from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformFatal(r, message=None):
    """Raises an InternalError exception that generates a stack trace.

  Args:
    r: A JSON-serializable object.
    message: An error message. If not specified then the resource is formatted
      as the error message.

  Raises:
    InternalError: This generates a stack trace.
  """
    raise resource_exceptions.InternalError(message if message is not None else six.text_type(r))