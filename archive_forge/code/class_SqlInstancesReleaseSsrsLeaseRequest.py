from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesReleaseSsrsLeaseRequest(_messages.Message):
    """A SqlInstancesReleaseSsrsLeaseRequest object.

  Fields:
    instance: Required. The Cloud SQL instance ID. This doesn't include the
      project ID. It's composed of lowercase letters, numbers, and hyphens,
      and it must start with a letter. The total length must be 98 characters
      or less (Example: instance-id).
    project: Required. The ID of the project that contains the instance
      (Example: project-id).
  """
    instance = _messages.StringField(1, required=True)
    project = _messages.StringField(2, required=True)