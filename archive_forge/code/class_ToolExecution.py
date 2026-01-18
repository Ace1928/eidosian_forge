from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolExecution(_messages.Message):
    """An execution of an arbitrary tool. It could be a test runner or a tool
  copying artifacts or deploying code.

  Fields:
    commandLineArguments: The full tokenized command line including the
      program name (equivalent to argv in a C program). - In response: present
      if set by create request - In create request: optional - In update
      request: never set
    exitCode: Tool execution exit code. This field will be set once the tool
      has exited. - In response: present if set by create/update request - In
      create request: optional - In update request: optional, a
      FAILED_PRECONDITION error will be returned if an exit_code is already
      set.
    toolLogs: References to any plain text logs output the tool execution.
      This field can be set before the tool has exited in order to be able to
      have access to a live view of the logs while the tool is running. The
      maximum allowed number of tool logs per step is 1000. - In response:
      present if set by create/update request - In create request: optional -
      In update request: optional, any value provided will be appended to the
      existing list
    toolOutputs: References to opaque files of any format output by the tool
      execution. The maximum allowed number of tool outputs per step is 1000.
      - In response: present if set by create/update request - In create
      request: optional - In update request: optional, any value provided will
      be appended to the existing list
  """
    commandLineArguments = _messages.StringField(1, repeated=True)
    exitCode = _messages.MessageField('ToolExitCode', 2)
    toolLogs = _messages.MessageField('FileReference', 3, repeated=True)
    toolOutputs = _messages.MessageField('ToolOutputReference', 4, repeated=True)