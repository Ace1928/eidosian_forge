from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import errno
import json
import random
import six
import gslib
from gslib.exception import CommandException
from gslib.tracker_file import (WriteJsonDataToTrackerFile,
from gslib.utils.constants import UTF8
def GenerateComponentObjectPrefix(encryption_key_sha256=None):
    """Generates a random prefix for component objects.

  Args:
    encryption_key_sha256: Encryption key SHA256 that will be used to encrypt
        the components. This is hashed into the prefix to avoid collision
        during resumption with a different encryption key.

  Returns:
    String prefix for use in the composite upload.
  """
    return str((random.randint(1, 10 ** 10 - 1) + hash(encryption_key_sha256)) % 10 ** 10)