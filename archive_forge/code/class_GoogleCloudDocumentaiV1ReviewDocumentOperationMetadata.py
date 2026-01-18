from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ReviewDocumentOperationMetadata(_messages.Message):
    """The long-running operation metadata for the ReviewDocument method.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
    questionId: The Crowd Compute question ID.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiV1CommonOperationMetadata', 1)
    questionId = _messages.StringField(2)