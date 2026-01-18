from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1MetadataStoreMetadataStoreState(_messages.Message):
    """Represents state information for a MetadataStore.

  Fields:
    diskUtilizationBytes: The disk utilization of the MetadataStore in bytes.
  """
    diskUtilizationBytes = _messages.IntegerField(1)