from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NFSMount(_messages.Message):
    """Configuration for an `NFSMount` to be attached to the VM.

  Fields:
    target: A target NFS mount. The target must be specified as
      `address:/mount".
  """
    target = _messages.StringField(1)