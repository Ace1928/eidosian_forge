from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gcloud_crcmod as crcmod
import six
def Crc32cMatches(data, data_crc32c):
    """Checks that the CRC32C checksum of the provided data matches the provided checksum.

  Args:
    data: bytes over which the checksum should be calculated.
    data_crc32c: int checksum against which data's checksum will be compared.

  Returns:
    True iff both checksums match.
  """
    return Crc32c(data) == data_crc32c