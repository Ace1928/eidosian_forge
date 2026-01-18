from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TerminateJobRunRequest(_messages.Message):
    """The request object used by `TerminateJobRun`.

  Fields:
    overrideDeployPolicy: Optional. Deploy policies to override. Format is
      `projects/{project}/locations/{location}/deployPolicies/a-z{0,62}`.
  """
    overrideDeployPolicy = _messages.StringField(1, repeated=True)