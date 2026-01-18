from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentOutputConfig(_messages.Message):
    """Config that controls the output of documents. All documents will be
  written as a JSON file.

  Fields:
    gcsOutputConfig: Output config to write the results to Cloud Storage.
  """
    gcsOutputConfig = _messages.MessageField('GoogleCloudDocumentaiV1DocumentOutputConfigGcsOutputConfig', 1)