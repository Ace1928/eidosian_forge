from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Fulfillment(_messages.Message):
    """By default, your agent responds to a matched intent with a static
  response. As an alternative, you can provide a more dynamic response by
  using fulfillment. When you enable fulfillment for an intent, Dialogflow
  responds to that intent by calling a service that you define. For example,
  if an end-user wants to schedule a haircut on Friday, your service can check
  your database and respond to the end-user with availability information for
  Friday. For more information, see the [fulfillment
  guide](https://cloud.google.com/dialogflow/docs/fulfillment-overview).

  Fields:
    displayName: Optional. The human-readable name of the fulfillment, unique
      within the agent. This field is not used for Fulfillment in an
      Environment.
    enabled: Optional. Whether fulfillment is enabled.
    features: Optional. The field defines whether the fulfillment is enabled
      for certain features.
    genericWebService: Configuration for a generic web service.
    name: Required. The unique identifier of the fulfillment. Supported
      formats: - `projects//agent/fulfillment` -
      `projects//locations//agent/fulfillment` This field is not used for
      Fulfillment in an Environment.
  """
    displayName = _messages.StringField(1)
    enabled = _messages.BooleanField(2)
    features = _messages.MessageField('GoogleCloudDialogflowV2FulfillmentFeature', 3, repeated=True)
    genericWebService = _messages.MessageField('GoogleCloudDialogflowV2FulfillmentGenericWebService', 4)
    name = _messages.StringField(5)