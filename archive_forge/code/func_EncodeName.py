from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from googlecloudsdk.core.cache import exceptions
import six
import six.moves.urllib.parse
@classmethod
def EncodeName(cls, name):
    """Returns name encoded for filesystem portability.

    A cache name may be a file path. The part after the rightmost of
    ('/', '\\\\') is encoded with Table.EncodeName().

    Args:
      name: The cache name string to encode.

    Raises:
      CacheNameInvalid: For invalid cache names.

    Returns:
      Name encoded for filesystem portability.
    """
    basename_index = max(name.rfind('/'), name.rfind('\\')) + 1
    if not name[basename_index:]:
        raise exceptions.CacheNameInvalid('Cache name [{}] is invalid.'.format(name))
    return name[:basename_index] + Table.EncodeName(name[basename_index:])