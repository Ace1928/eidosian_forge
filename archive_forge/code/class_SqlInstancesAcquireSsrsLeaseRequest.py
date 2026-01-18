from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesAcquireSsrsLeaseRequest(_messages.Message):
    """A SqlInstancesAcquireSsrsLeaseRequest object.

  Fields:
    instance: Required. Cloud SQL instance ID. This doesn't include the
      project ID. It's composed of lowercase letters, numbers, and hyphens,
      and it must start with a letter. The total length must be 98 characters
      or less (Example: instance-id).
    instancesAcquireSsrsLeaseRequest: A InstancesAcquireSsrsLeaseRequest
      resource to be passed as the request body.
    project: Required. ID of the project that contains the instance (Example:
      project-id).
  """
    instance = _messages.StringField(1, required=True)
    instancesAcquireSsrsLeaseRequest = _messages.MessageField('InstancesAcquireSsrsLeaseRequest', 2)
    project = _messages.StringField(3, required=True)