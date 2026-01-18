from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import random
import string
import six
from six.moves import range  # pylint: disable=redefined-builtin
def GenerateRandomName():
    """Generates a random string.

  Returns:
    The returned string will be 12 characters long and will begin with
    a lowercase letter followed by 11 characters drawn from the set
    [a-z0-9].
  """
    buf = io.StringIO()
    buf.write(six.text_type(random.choice(_BEGIN_ALPHABET)))
    for _ in range(_LENGTH - 1):
        buf.write(six.text_type(random.choice(_ALPHABET)))
    return buf.getvalue()