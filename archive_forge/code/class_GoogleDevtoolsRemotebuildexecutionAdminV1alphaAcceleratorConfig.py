from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaAcceleratorConfig(_messages.Message):
    """AcceleratorConfig defines the accelerator cards to attach to the VM.

  Fields:
    acceleratorCount: The number of guest accelerator cards exposed to each
      VM.
    acceleratorType: The type of accelerator to attach to each VM, e.g.
      "nvidia-tesla-k80" for nVidia Tesla K80.
  """
    acceleratorCount = _messages.IntegerField(1)
    acceleratorType = _messages.StringField(2)