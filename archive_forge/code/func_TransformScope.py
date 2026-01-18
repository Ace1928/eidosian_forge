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
def TransformScope(r, *args):
    """Gets the /args/ suffix from a URI.

  Args:
    r: A URI.
    *args: Optional URI segment names. If not specified then 'regions', 'zones'
      is assumed.

  Returns:
    The URI segment after the first /*args/ in r, the last /-separated
      component in r if none found.

  Example:
    `"http://abc/foo/projects/bar/xyz".scope("projects")` returns `"bar/xyz"`.

    `"http://xyz/foo/regions/abc".scope()` returns `"abc"`.
  """
    if not r:
        return ''
    r = urllib.parse.unquote(six.text_type(r))
    if '/' not in r:
        return r
    for scope in args or ('regions', 'zones'):
        segment = '/' + scope + '/'
        if segment in r:
            return r.split(segment)[-1]
    if r.startswith('https://'):
        return r.split('/')[-1]
    return r