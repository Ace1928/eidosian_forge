from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3WebhookServiceDirectoryConfig(_messages.Message):
    """Represents configuration for a [Service
  Directory](https://cloud.google.com/service-directory) service.

  Fields:
    genericWebService: Generic Service configuration of this webhook.
    service: Required. The name of [Service
      Directory](https://cloud.google.com/service-directory) service. Format:
      `projects//locations//namespaces//services/`. `Location ID` of the
      service directory must be the same as the location of the agent.
  """
    genericWebService = _messages.MessageField('GoogleCloudDialogflowCxV3WebhookGenericWebService', 1)
    service = _messages.StringField(2)