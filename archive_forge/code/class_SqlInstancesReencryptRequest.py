from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesReencryptRequest(_messages.Message):
    """A SqlInstancesReencryptRequest object.

  Fields:
    instance: Cloud SQL instance ID. This does not include the project ID.
    instancesReencryptRequest: A InstancesReencryptRequest resource to be
      passed as the request body.
    project: ID of the project that contains the instance.
  """
    instance = _messages.StringField(1, required=True)
    instancesReencryptRequest = _messages.MessageField('InstancesReencryptRequest', 2)
    project = _messages.StringField(3, required=True)