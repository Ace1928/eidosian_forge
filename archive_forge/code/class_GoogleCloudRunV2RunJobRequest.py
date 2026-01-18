from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2RunJobRequest(_messages.Message):
    """Request message to create a new Execution of a Job.

  Fields:
    etag: A system-generated fingerprint for this version of the resource. May
      be used to detect modification conflict during updates.
    overrides: Overrides specification for a given execution of a job. If
      provided, overrides will be applied to update the execution or task
      spec.
    validateOnly: Indicates that the request should be validated without
      actually deleting any resources.
  """
    etag = _messages.StringField(1)
    overrides = _messages.MessageField('GoogleCloudRunV2Overrides', 2)
    validateOnly = _messages.BooleanField(3)