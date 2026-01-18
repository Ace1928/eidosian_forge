from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1FlowHook(_messages.Message):
    """A GoogleCloudApigeeV1FlowHook object.

  Fields:
    continueOnError: Optional. Flag that specifies whether execution should
      continue if the flow hook throws an exception. Set to `true` to continue
      execution. Set to `false` to stop execution if the flow hook throws an
      exception. Defaults to `true`.
    description: Description of the flow hook.
    flowHookPoint: Output only. Where in the API call flow the flow hook is
      invoked. Must be one of `PreProxyFlowHook`, `PostProxyFlowHook`,
      `PreTargetFlowHook`, or `PostTargetFlowHook`.
    sharedFlow: Shared flow attached to this flow hook, or empty if there is
      none attached.
  """
    continueOnError = _messages.BooleanField(1)
    description = _messages.StringField(2)
    flowHookPoint = _messages.StringField(3)
    sharedFlow = _messages.StringField(4)