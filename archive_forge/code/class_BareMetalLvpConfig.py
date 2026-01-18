from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalLvpConfig(_messages.Message):
    """Specifies the configs for local persistent volumes (PVs).

  Fields:
    path: Required. The host machine path.
    storageClass: Required. The StorageClass name that PVs will be created
      with.
  """
    path = _messages.StringField(1)
    storageClass = _messages.StringField(2)