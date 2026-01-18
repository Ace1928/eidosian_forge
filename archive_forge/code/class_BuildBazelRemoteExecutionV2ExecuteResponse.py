from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BuildBazelRemoteExecutionV2ExecuteResponse(_messages.Message):
    """The response message for Execution.Execute, which will be contained in
  the response field of the Operation.

  Messages:
    ServerLogsValue: An optional list of additional log outputs the server
      wishes to provide. A server can use this to return execution-specific
      logs however it wishes. This is intended primarily to make it easier for
      users to debug issues that may be outside of the actual job execution,
      such as by identifying the worker executing the action or by providing
      logs from the worker's setup phase. The keys SHOULD be human readable so
      that a client can display them to a user.

  Fields:
    cachedResult: True if the result was served from cache, false if it was
      executed.
    message: Freeform informational message with details on the execution of
      the action that may be displayed to the user upon failure or when
      requested explicitly.
    result: The result of the action.
    serverLogs: An optional list of additional log outputs the server wishes
      to provide. A server can use this to return execution-specific logs
      however it wishes. This is intended primarily to make it easier for
      users to debug issues that may be outside of the actual job execution,
      such as by identifying the worker executing the action or by providing
      logs from the worker's setup phase. The keys SHOULD be human readable so
      that a client can display them to a user.
    status: If the status has a code other than `OK`, it indicates that the
      action did not finish execution. For example, if the operation times out
      during execution, the status will have a `DEADLINE_EXCEEDED` code.
      Servers MUST use this field for errors in execution, rather than the
      error field on the `Operation` object. If the status code is other than
      `OK`, then the result MUST NOT be cached. For an error status, the
      `result` field is optional; the server may populate the output-,
      stdout-, and stderr-related fields if it has any information available,
      such as the stdout and stderr of a timed-out action.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ServerLogsValue(_messages.Message):
        """An optional list of additional log outputs the server wishes to
    provide. A server can use this to return execution-specific logs however
    it wishes. This is intended primarily to make it easier for users to debug
    issues that may be outside of the actual job execution, such as by
    identifying the worker executing the action or by providing logs from the
    worker's setup phase. The keys SHOULD be human readable so that a client
    can display them to a user.

    Messages:
      AdditionalProperty: An additional property for a ServerLogsValue object.

    Fields:
      additionalProperties: Additional properties of type ServerLogsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ServerLogsValue object.

      Fields:
        key: Name of the additional property.
        value: A BuildBazelRemoteExecutionV2LogFile attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('BuildBazelRemoteExecutionV2LogFile', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cachedResult = _messages.BooleanField(1)
    message = _messages.StringField(2)
    result = _messages.MessageField('BuildBazelRemoteExecutionV2ActionResult', 3)
    serverLogs = _messages.MessageField('ServerLogsValue', 4)
    status = _messages.MessageField('GoogleRpcStatus', 5)