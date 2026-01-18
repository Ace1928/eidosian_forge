from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
import unicodedata
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import times
import six
def _Stringize(value):
    """Returns the unicode string representation for value."""
    if value is None:
        return 'null'
    if not isinstance(value, six.string_types):
        value = repr(value)
    return six.text_type(encoding.Decode(value))