from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedValues(_messages.Message):
    """SupportedValues represents the values supported by the configuration.

  Fields:
    acceleratorTypes: Output only. The accelerator types supported by WbI.
    machineTypes: Output only. The machine types supported by WbI.
  """
    acceleratorTypes = _messages.StringField(1, repeated=True)
    machineTypes = _messages.StringField(2, repeated=True)