from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTestCasesResponse(_messages.Message):
    """Response message for StepService.ListTestCases.

  Fields:
    nextPageToken: A string attribute.
    testCases: List of test cases.
  """
    nextPageToken = _messages.StringField(1)
    testCases = _messages.MessageField('TestCase', 2, repeated=True)