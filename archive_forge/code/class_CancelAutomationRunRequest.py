from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CancelAutomationRunRequest(_messages.Message):
    """The request object used by `CancelAutomationRun`.

  Fields:
    overrideDeployPolicy: Deploy policies to override. Format is
      `projects/{project}/
      locations/{location}/deployPolicies/{deploy_policy}`.
  """
    overrideDeployPolicy = _messages.StringField(1, repeated=True)