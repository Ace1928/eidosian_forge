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
def TransformList(r, show='', undefined='', separator=','):
    """Formats a dict or list into a compact comma separated list.

  Args:
    r: A JSON-serializable object.
    show: If show=*keys* then list dict keys; if show=*values* then list dict
      values; otherwise list dict key=value pairs.
    undefined: Return this if the resource is empty.
    separator: The list item separator string.

  Returns:
    The key=value pairs for a dict or list values for a list, separated by
    separator. Returns undefined if r is empty, or r if it is not a dict or
    list.
  """
    if isinstance(r, dict):
        if show == 'keys':
            return separator.join([six.text_type(k) for k in sorted(r)])
        elif show == 'values':
            return separator.join([six.text_type(v) for _, v in sorted(six.iteritems(r))])
        else:
            return separator.join(['{k}={v}'.format(k=k, v=v) for k, v in sorted(six.iteritems(r))])
    if isinstance(r, list):
        return separator.join(map(six.text_type, r))
    return r or undefined