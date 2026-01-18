from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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