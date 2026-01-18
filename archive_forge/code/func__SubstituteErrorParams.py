from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def _SubstituteErrorParams(fmt, params):
    """Replaces $N with the Nth param in fmt.

  Args:
    fmt: A format string which may contain substitutions of the form $N, where
      N is any decimal integer between 0 and len(params) - 1.
    params: A set of parameters to substitute in place of the $N string.
  Returns:
    A string containing fmt with each $N substring replaced with its
    corresponding parameter.
  """
    if not params:
        return fmt
    return re.sub('\\$([0-9]+)', '{\\1}', fmt).format(*params)