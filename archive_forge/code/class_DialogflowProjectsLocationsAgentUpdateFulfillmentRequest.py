from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentUpdateFulfillmentRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentUpdateFulfillmentRequest object.

  Fields:
    googleCloudDialogflowV2Fulfillment: A GoogleCloudDialogflowV2Fulfillment
      resource to be passed as the request body.
    name: Required. The unique identifier of the fulfillment. Supported
      formats: - `projects//agent/fulfillment` -
      `projects//locations//agent/fulfillment` This field is not used for
      Fulfillment in an Environment.
    updateMask: Required. The mask to control which fields get updated. If the
      mask is not present, all fields will be updated.
  """
    googleCloudDialogflowV2Fulfillment = _messages.MessageField('GoogleCloudDialogflowV2Fulfillment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)