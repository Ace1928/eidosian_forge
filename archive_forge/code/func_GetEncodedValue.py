from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import six
def GetEncodedValue(env, name, default=None):
    """Returns the decoded value of the env var name.

  Args:
    env: {str: str}, The env dict.
    name: str, The env var name.
    default: The value to return if name is not in env.

  Returns:
    The decoded value of the env var name.
  """
    name = Encode(name)
    value = env.get(name)
    if value is None:
        return default
    return Decode(value)