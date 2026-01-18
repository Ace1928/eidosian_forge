from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesProjectSettingsPatchRequest(_messages.Message):
    """A ServicemanagementServicesProjectSettingsPatchRequest object.

  Fields:
    consumerProjectId: The project ID of the consumer.
    projectSettings: A ProjectSettings resource to be passed as the request
      body.
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`.
    updateMask: The field mask specifying which fields are to be updated.
  """
    consumerProjectId = _messages.StringField(1, required=True)
    projectSettings = _messages.MessageField('ProjectSettings', 2)
    serviceName = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)