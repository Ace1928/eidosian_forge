from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesConfigsSubmitRequest(_messages.Message):
    """A ServicemanagementServicesConfigsSubmitRequest object.

  Fields:
    serviceName: The name of the service.  See the `ServiceManager` overview
      for naming requirements.  For example: `example.googleapis.com`.
    submitConfigSourceRequest: A SubmitConfigSourceRequest resource to be
      passed as the request body.
  """
    serviceName = _messages.StringField(1, required=True)
    submitConfigSourceRequest = _messages.MessageField('SubmitConfigSourceRequest', 2)