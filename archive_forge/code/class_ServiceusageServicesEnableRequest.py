from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesEnableRequest(_messages.Message):
    """A ServiceusageServicesEnableRequest object.

  Fields:
    enableServiceRequest: A EnableServiceRequest resource to be passed as the
      request body.
    name: Name of the consumer and service to enable the service on. The
      `EnableService` and `DisableService` methods currently only support
      projects. Enabling a service requires that the service is public or is
      shared with the user enabling the service. An example name would be:
      `projects/123/services/serviceusage.googleapis.com` where `123` is the
      project number.
  """
    enableServiceRequest = _messages.MessageField('EnableServiceRequest', 1)
    name = _messages.StringField(2, required=True)