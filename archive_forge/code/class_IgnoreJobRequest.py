from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IgnoreJobRequest(_messages.Message):
    """The request object used by `IgnoreJob`.

  Fields:
    jobId: Required. The job ID for the Job to ignore.
    overrideDeployPolicy: Optional. Deploy policies to override. Format is
      `projects/{project}/locations/{location}/deployPolicies/a-z{0,62}`.
    phaseId: Required. The phase ID the Job to ignore belongs to.
  """
    jobId = _messages.StringField(1)
    overrideDeployPolicy = _messages.StringField(2, repeated=True)
    phaseId = _messages.StringField(3)