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
def TransformEncode(r, encoding, undefined=''):
    """Returns the encoded value of the resource using encoding.

  Args:
    r: A JSON-serializable object.
    encoding: The encoding name. *base64* and *utf-8* are supported.
    undefined: Returns this value if the encoding fails.

  Returns:
    The encoded resource.
  """
    if encoding == 'base64':
        try:
            b = base64.b64encode(console_attr.EncodeToBytes(r))
            return console_attr.SafeText(b).rstrip('\n')
        except:
            return undefined
    try:
        return console_attr.SafeText(r, encoding)
    except:
        return undefined