from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1TrainProcessorVersionRequestInputData(_messages.Message):
    """The input data used to train a new ProcessorVersion.

  Fields:
    testDocuments: The documents used for testing the trained version.
    trainingDocuments: The documents used for training the new version.
  """
    testDocuments = _messages.MessageField('GoogleCloudDocumentaiV1BatchDocumentsInputConfig', 1)
    trainingDocuments = _messages.MessageField('GoogleCloudDocumentaiV1BatchDocumentsInputConfig', 2)