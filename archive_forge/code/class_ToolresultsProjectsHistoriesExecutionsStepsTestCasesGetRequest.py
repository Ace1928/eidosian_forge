from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsTestCasesGetRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsStepsTestCasesGetRequest object.

  Fields:
    executionId: A Execution id Required.
    historyId: A History id. Required.
    projectId: A Project id. Required.
    stepId: A Step id. Note: This step must include a TestExecutionStep.
      Required.
    testCaseId: A Test Case id. Required.
  """
    executionId = _messages.StringField(1, required=True)
    historyId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    stepId = _messages.StringField(4, required=True)
    testCaseId = _messages.StringField(5, required=True)