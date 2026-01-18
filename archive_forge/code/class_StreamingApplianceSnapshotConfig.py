from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingApplianceSnapshotConfig(_messages.Message):
    """Streaming appliance snapshot configuration.

  Fields:
    importStateEndpoint: Indicates which endpoint is used to import appliance
      state.
    snapshotId: If set, indicates the snapshot id for the snapshot being
      performed.
  """
    importStateEndpoint = _messages.StringField(1)
    snapshotId = _messages.StringField(2)