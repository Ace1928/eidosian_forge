from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ExportImageResponse(_messages.Message):
    """ExportImageResponse contains an operation Id to track the image export
  operation.

  Fields:
    operationId: An operation ID used to track the status of image exports
      tied to the original pod ID in the request.
  """
    operationId = _messages.StringField(1)