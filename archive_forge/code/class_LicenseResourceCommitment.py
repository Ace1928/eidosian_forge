from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LicenseResourceCommitment(_messages.Message):
    """Commitment for a particular license resource.

  Fields:
    amount: The number of licenses purchased.
    coresPerLicense: Specifies the core range of the instance for which this
      license applies.
    license: Any applicable license URI.
  """
    amount = _messages.IntegerField(1)
    coresPerLicense = _messages.StringField(2)
    license = _messages.StringField(3)