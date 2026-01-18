from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesExportRequest(_messages.Message):
    """A SqlInstancesExportRequest object.

  Fields:
    instance: The Cloud SQL instance ID. This doesn't include the project ID.
    instancesExportRequest: A InstancesExportRequest resource to be passed as
      the request body.
    project: Project ID of the project that contains the instance to be
      exported.
  """
    instance = _messages.StringField(1, required=True)
    instancesExportRequest = _messages.MessageField('InstancesExportRequest', 2)
    project = _messages.StringField(3, required=True)