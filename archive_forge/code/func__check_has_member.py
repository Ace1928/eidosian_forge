import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def _check_has_member(self, member, error_msg):
    """Check that the object has a given member variable.

        Args:
          member: the member variable to check for
          error_msg: the message for an error if the member is missing
        Raises:
          ObjectFormatException: with the given error_msg if member is
            missing or is None
        """
    if getattr(self, member, None) is None:
        raise ObjectFormatException(error_msg)