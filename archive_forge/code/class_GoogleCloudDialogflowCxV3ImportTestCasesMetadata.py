from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3ImportTestCasesMetadata(_messages.Message):
    """Metadata returned for the TestCases.ImportTestCases long running
  operation.

  Fields:
    errors: Errors for failed test cases.
  """
    errors = _messages.MessageField('GoogleCloudDialogflowCxV3TestCaseError', 1, repeated=True)