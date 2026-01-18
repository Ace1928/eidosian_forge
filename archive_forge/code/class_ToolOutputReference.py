from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolOutputReference(_messages.Message):
    """A reference to a ToolExecution output file.

  Fields:
    creationTime: The creation time of the file. - In response: present if set
      by create/update request - In create/update request: optional
    output: A FileReference to an output file. - In response: always set - In
      create/update request: always set
    testCase: The test case to which this output file belongs. - In response:
      present if set by create/update request - In create/update request:
      optional
  """
    creationTime = _messages.MessageField('Timestamp', 1)
    output = _messages.MessageField('FileReference', 2)
    testCase = _messages.MessageField('TestCaseReference', 3)