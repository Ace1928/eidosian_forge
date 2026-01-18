from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ExportTestCasesResponse(_messages.Message):
    """The response message for TestCases.ExportTestCases.

  Fields:
    content: Uncompressed raw byte content for test cases.
    gcsUri: The URI to a file containing the exported test cases. This field
      is populated only if `gcs_uri` is specified in ExportTestCasesRequest.
  """
    content = _messages.BytesField(1)
    gcsUri = _messages.StringField(2)