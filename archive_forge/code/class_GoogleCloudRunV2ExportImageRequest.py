from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ExportImageRequest(_messages.Message):
    """Request message for exporting Cloud Run image.

  Fields:
    destinationRepo: Required. The export destination url (the Artifact
      Registry repo).
  """
    destinationRepo = _messages.StringField(1)