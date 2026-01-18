from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3TrainProcessorVersionResponse(_messages.Message):
    """The response for TrainProcessorVersion.

  Fields:
    processorVersion: The resource name of the processor version produced by
      training.
  """
    processorVersion = _messages.StringField(1)