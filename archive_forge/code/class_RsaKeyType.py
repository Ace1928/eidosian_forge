from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RsaKeyType(_messages.Message):
    """Describes an RSA key that may be used in a Certificate issued from a
  CaPool.

  Fields:
    maxModulusSize: Optional. The maximum allowed RSA modulus size
      (inclusive), in bits. If this is not set, or if set to zero, the service
      will not enforce an explicit upper bound on RSA modulus sizes.
    minModulusSize: Optional. The minimum allowed RSA modulus size
      (inclusive), in bits. If this is not set, or if set to zero, the
      service-level min RSA modulus size will continue to apply.
  """
    maxModulusSize = _messages.IntegerField(1)
    minModulusSize = _messages.IntegerField(2)