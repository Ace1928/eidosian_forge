from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3DisableProcessorMetadata(_messages.Message):
    """The long-running operation metadata for the DisableProcessor method.

  Fields:
    commonMetadata: The basic metadata of the long-running operation.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiV1beta3CommonOperationMetadata', 1)