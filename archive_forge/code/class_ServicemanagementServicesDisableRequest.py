from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesDisableRequest(_messages.Message):
    """A ServicemanagementServicesDisableRequest object.

  Fields:
    disableServiceRequest: A DisableServiceRequest resource to be passed as
      the request body.
    serviceName: Name of the service to disable. Specifying an unknown service
      name will cause the request to fail.
  """
    disableServiceRequest = _messages.MessageField('DisableServiceRequest', 1)
    serviceName = _messages.StringField(2, required=True)