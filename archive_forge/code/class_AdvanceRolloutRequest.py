from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdvanceRolloutRequest(_messages.Message):
    """The request object used by `AdvanceRollout`.

  Fields:
    overrideDeployPolicy: Optional. Deploy policies to override. Format is
      `projects/{project}/locations/{location}/deployPolicies/a-z{0,62}`.
    phaseId: Required. The phase ID to advance the `Rollout` to.
  """
    overrideDeployPolicy = _messages.StringField(1, repeated=True)
    phaseId = _messages.StringField(2)