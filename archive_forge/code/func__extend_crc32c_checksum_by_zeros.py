from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def _extend_crc32c_checksum_by_zeros(crc_checksum, bit_count):
    """Given crc_checksum representing polynomial P(x), compute P(x)*x^bit_count.

  Args:
    crc_checksum (int): crc respresenting polynomial P(x).
    bit_count (int): number of bits in crc.

  Returns:
    P(x)*x^bit_count (int).
  """
    updated_crc_checksum = _reverse_32_bits(crc_checksum)
    i = 0
    while bit_count != 0:
        if bit_count & 1:
            updated_crc_checksum = _multiply_crc_polynomials(updated_crc_checksum, X_POW_2K_TABLE[i % len(X_POW_2K_TABLE)])
        i += 1
        bit_count >>= 1
    updated_crc_checksum = _reverse_32_bits(updated_crc_checksum)
    return updated_crc_checksum