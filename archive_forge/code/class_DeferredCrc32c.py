from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import struct
import textwrap
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class DeferredCrc32c(object):
    """Hashlib-like helper for deferring hash calculations to gcloud-crc32c.

  NOTE: Given this class relies on analyzing data on disk, it is not appropriate
  for hashing streaming downloads and will fail to work as expected.
  """

    def __init__(self, crc=0):
        """Sets up the internal checksum variable and allows an initial value.

    Args:
      crc (int): The initial checksum to be stored.
    """
        self._crc = crc

    def copy(self):
        return DeferredCrc32c(crc=self._crc)

    def update(self, data):
        del data
        return

    def sum_file(self, file_path, offset, length):
        """Calculates checksum on a provided file path.

    Args:
      file_path (str): A string representing a path to a file.
      offset (int): The number of bytes to offset from the beginning of the
        file. Defaults to 0.
      length (int): The number of bytes to read into the file. If not specified
        will calculate until the end of file is encountered.
    """
        if offset is None or length is None:
            raise errors.Error('gcloud_crc32c binary uses 0 (not `None`) to indicate "no argument given."')
        crc32c_operation = GcloudCrc32cOperation()
        result = crc32c_operation(file_path=file_path, offset=offset, length=length)
        self._crc = 0 if result.failed else int(result.stdout)

    def digest(self):
        """Returns the checksum in big-endian order, per RFC 4960.

    See: https://cloud.google.com/storage/docs/json_api/v1/objects#crc32c

    Returns:
      An eight-byte digest string.
    """
        return struct.pack('>L', self._crc)

    def hexdigest(self):
        """Returns a checksum like `digest` except as a bytestring of double length.

    Returns:
      A sixteen byte digest string, containing only hex digits.
    """
        return '{:08x}'.format(self._crc).encode('ascii')