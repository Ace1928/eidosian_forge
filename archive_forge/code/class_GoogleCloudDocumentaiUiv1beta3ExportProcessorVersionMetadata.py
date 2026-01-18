from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiUiv1beta3ExportProcessorVersionMetadata(_messages.Message):
    """Metadata message associated with the ExportProcessorVersion operation.

  Fields:
    commonMetadata: The common metadata about the operation.
  """
    commonMetadata = _messages.MessageField('GoogleCloudDocumentaiUiv1beta3CommonOperationMetadata', 1)