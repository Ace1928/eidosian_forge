from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DateShiftConfig(_messages.Message):
    """Shifts a date forward or backward in time by a random amount that is
  consistent for a given patient and crypto key combination.

  Fields:
    cryptoKey: An AES 128/192/256 bit key. The date shift is computed based on
      this key and the patient ID. If the patient ID is empty for a DICOM
      resource, the date shift is computed based on this key and the study
      instance UID. If crypto_key is not set, then kms_wrapped is used to
      calculate the date shift. If neither is set, a default key is generated
      for each de-identify operation. Must not be set if kms_wrapped is set.
  """
    cryptoKey = _messages.BytesField(1)