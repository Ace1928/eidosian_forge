from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pkgutil
import textwrap
from googlecloudsdk import api_lib
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _StripUnusedNotation(string):
    """Returns string with Pythonic unused notation stripped."""
    if string.startswith('_'):
        return string.lstrip('_')
    unused = 'unused_'
    if string.startswith(unused):
        return string[len(unused):]
    return string