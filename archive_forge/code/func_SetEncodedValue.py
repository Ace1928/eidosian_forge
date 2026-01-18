from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import six
def SetEncodedValue(env, name, value, encoding=None):
    """Sets the value of name in env to an encoded value.

  Args:
    env: {str: str}, The env dict.
    name: str, The env var name.
    value: str or unicode, The value for name. If None then name is removed from
      env.
    encoding: str, The encoding to use or None to try to infer it.
  """
    name = Encode(name, encoding=encoding)
    if value is None:
        env.pop(name, None)
        return
    env[name] = Encode(value, encoding=encoding)