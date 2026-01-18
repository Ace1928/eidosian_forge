from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1FetchProcessorTypesResponse(_messages.Message):
    """Response message for the FetchProcessorTypes method.

  Fields:
    processorTypes: The list of processor types.
  """
    processorTypes = _messages.MessageField('GoogleCloudDocumentaiV1ProcessorType', 1, repeated=True)