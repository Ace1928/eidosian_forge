from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3TestCase(_messages.Message):
    """Represents a test case.

  Fields:
    creationTime: Output only. When the test was created.
    displayName: Required. The human-readable name of the test case, unique
      within the agent. Limit of 200 characters.
    lastTestResult: The latest test result.
    name: The unique identifier of the test case. TestCases.CreateTestCase
      will populate the name automatically. Otherwise use format:
      `projects//locations//agents/ /testCases/`.
    notes: Additional freeform notes about the test case. Limit of 400
      characters.
    tags: Tags are short descriptions that users may apply to test cases for
      organizational and filtering purposes. Each tag should start with "#"
      and has a limit of 30 characters.
    testCaseConversationTurns: The conversation turns uttered when the test
      case was created, in chronological order. These include the canonical
      set of agent utterances that should occur when the agent is working
      properly.
    testConfig: Config for the test case.
  """
    creationTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    lastTestResult = _messages.MessageField('GoogleCloudDialogflowCxV3TestCaseResult', 3)
    name = _messages.StringField(4)
    notes = _messages.StringField(5)
    tags = _messages.StringField(6, repeated=True)
    testCaseConversationTurns = _messages.MessageField('GoogleCloudDialogflowCxV3ConversationTurn', 7, repeated=True)
    testConfig = _messages.MessageField('GoogleCloudDialogflowCxV3TestConfig', 8)