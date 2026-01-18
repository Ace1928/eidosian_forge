from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta3ImportProcessorVersionResponse(_messages.Message):
    """The response message for the ImportProcessorVersion method.

  Fields:
    processorVersion: The destination processor version name.
  """
    processorVersion = _messages.StringField(1)