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
def GetTreeSizeBytes(path, predicate=None):
    """Returns sum of sizes of not-ignored files under given path, in bytes."""
    result = 0
    if predicate is None:
        predicate = lambda x: True
    for directory in os.walk(six.text_type(path)):
        for file_name in directory[2]:
            file_path = os.path.join(directory[0], file_name)
            if predicate(file_path):
                result += os.path.getsize(file_path)
    return result