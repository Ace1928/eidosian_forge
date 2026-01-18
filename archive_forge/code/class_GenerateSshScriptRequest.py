from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateSshScriptRequest(_messages.Message):
    """Request message for 'GenerateSshScript' request.

  Fields:
    vm: Required. Bastion VM Instance name to use or to create.
    vmCreationConfig: The VM creation configuration.
    vmPort: The port that will be open on the bastion host
    vmSelectionConfig: The VM selection configuration.
  """
    vm = _messages.StringField(1)
    vmCreationConfig = _messages.MessageField('VmCreationConfig', 2)
    vmPort = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    vmSelectionConfig = _messages.MessageField('VmSelectionConfig', 4)