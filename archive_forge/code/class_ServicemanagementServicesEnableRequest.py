from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicemanagementServicesEnableRequest(_messages.Message):
    """A ServicemanagementServicesEnableRequest object.

  Fields:
    enableServiceRequest: A EnableServiceRequest resource to be passed as the
      request body.
    serviceName: Name of the service to enable. Specifying an unknown service
      name will cause the request to fail.
  """
    enableServiceRequest = _messages.MessageField('EnableServiceRequest', 1)
    serviceName = _messages.StringField(2, required=True)