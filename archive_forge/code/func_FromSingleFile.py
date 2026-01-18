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
@staticmethod
def FromSingleFile(input_path, algorithm=hashlib.sha256):
    """Creates a Checksum containing one file.

    Args:
      input_path: str, The file path of the contents to add.
      algorithm: a hashing algorithm method, a la hashlib.algorithms

    Returns:
      Checksum, The checksum containing the file.
    """
    return Checksum(algorithm=algorithm).AddFileContents(input_path)