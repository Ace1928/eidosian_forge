from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FormFactorValueValuesEnum(_messages.Enum):
    """Whether this device is a phone, tablet, wearable, etc.

    Values:
      DEVICE_FORM_FACTOR_UNSPECIFIED: Do not use. For proto versioning only.
      PHONE: This device has the shape of a phone.
      TABLET: This device has the shape of a tablet.
      WEARABLE: This device has the shape of a watch or other wearable.
    """
    DEVICE_FORM_FACTOR_UNSPECIFIED = 0
    PHONE = 1
    TABLET = 2
    WEARABLE = 3