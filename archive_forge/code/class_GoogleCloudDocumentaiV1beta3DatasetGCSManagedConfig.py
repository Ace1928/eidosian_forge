from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3DatasetGCSManagedConfig(_messages.Message):
    """Configuration specific to the Cloud Storage-based implementation.

  Fields:
    gcsPrefix: Required. The Cloud Storage URI (a directory) where the
      documents belonging to the dataset must be stored.
  """
    gcsPrefix = _messages.MessageField('GoogleCloudDocumentaiV1beta3GcsPrefix', 1)