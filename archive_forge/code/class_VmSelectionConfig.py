from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmSelectionConfig(_messages.Message):
    """VM selection configuration message

  Fields:
    vmZone: Required. The Google Cloud Platform zone the VM is located.
  """
    vmZone = _messages.StringField(1)