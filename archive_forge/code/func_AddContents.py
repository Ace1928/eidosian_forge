from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import errno
import hashlib
import io
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding as encoding_util
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves import range  # pylint: disable=redefined-builtin
def AddContents(self, contents):
    """Adds the given contents to the checksum.

    Args:
      contents: str or bytes, The contents to add.

    Returns:
      self, For method chaining.
    """
    self.__hash.update(six.ensure_binary(contents))
    return self