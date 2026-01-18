from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetachLunRequest(_messages.Message):
    """Message for detach specific LUN from an Instance.

  Fields:
    lun: Required. Name of the Lun to detach.
    skipReboot: If true, performs lun unmapping without instance reboot.
  """
    lun = _messages.StringField(1)
    skipReboot = _messages.BooleanField(2)