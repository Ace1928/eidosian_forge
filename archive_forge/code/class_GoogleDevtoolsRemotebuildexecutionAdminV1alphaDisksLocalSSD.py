from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaDisksLocalSSD(_messages.Message):
    """LocalSSD specifies how to attach local SSD to the workers.

  Fields:
    count: Optional. The number of Local SSDs to be attached.
    sizeGb: Output only. The size of the local SSD in gb. Intended for
      informational purposes only.
  """
    count = _messages.IntegerField(1)
    sizeGb = _messages.IntegerField(2)