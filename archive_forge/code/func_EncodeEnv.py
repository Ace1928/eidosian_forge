from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import six
def EncodeEnv(env, encoding=None):
    """Encodes all the key value pairs in env in preparation for subprocess.

  Args:
    env: {str: str}, The environment you are going to pass to subprocess.
    encoding: str, The encoding to use or None to use the default.

  Returns:
    {bytes: bytes}, The environment to pass to subprocess.
  """
    encoding = encoding or _GetEncoding()
    encoded_env = {}
    for k, v in six.iteritems(env):
        SetEncodedValue(encoded_env, k, v, encoding=encoding)
    return encoded_env