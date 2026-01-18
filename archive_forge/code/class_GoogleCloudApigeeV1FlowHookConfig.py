from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1FlowHookConfig(_messages.Message):
    """A GoogleCloudApigeeV1FlowHookConfig object.

  Fields:
    continueOnError: Flag that specifies whether the flow should abort after
      an error in the flow hook. Defaults to `true` (continue on error).
    name: Name of the flow hook in the following format:
      `organizations/{org}/environments/{env}/flowhooks/{point}`. Valid
      `point` values include: `PreProxyFlowHook`, `PostProxyFlowHook`,
      `PreTargetFlowHook`, and `PostTargetFlowHook`
    sharedFlowName: Name of the shared flow to invoke in the following format:
      `organizations/{org}/sharedflows/{sharedflow}`
  """
    continueOnError = _messages.BooleanField(1)
    name = _messages.StringField(2)
    sharedFlowName = _messages.StringField(3)