from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DeployModelRequest(_messages.Message):
    """Request message for EndpointService.DeployModel.

  Messages:
    TrafficSplitValue: A map from a DeployedModel's ID to the percentage of
      this Endpoint's traffic that should be forwarded to that DeployedModel.
      If this field is non-empty, then the Endpoint's traffic_split will be
      overwritten with it. To refer to the ID of the just being deployed
      Model, a "0" should be used, and the actual ID of the new DeployedModel
      will be filled in its place by this method. The traffic percentage
      values must add up to 100. If this field is empty, then the Endpoint's
      traffic_split is not updated.

  Fields:
    deployedModel: Required. The DeployedModel to be created within the
      Endpoint. Note that Endpoint.traffic_split must be updated for the
      DeployedModel to start receiving traffic, either as part of this call,
      or via EndpointService.UpdateEndpoint.
    trafficSplit: A map from a DeployedModel's ID to the percentage of this
      Endpoint's traffic that should be forwarded to that DeployedModel. If
      this field is non-empty, then the Endpoint's traffic_split will be
      overwritten with it. To refer to the ID of the just being deployed
      Model, a "0" should be used, and the actual ID of the new DeployedModel
      will be filled in its place by this method. The traffic percentage
      values must add up to 100. If this field is empty, then the Endpoint's
      traffic_split is not updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TrafficSplitValue(_messages.Message):
        """A map from a DeployedModel's ID to the percentage of this Endpoint's
    traffic that should be forwarded to that DeployedModel. If this field is
    non-empty, then the Endpoint's traffic_split will be overwritten with it.
    To refer to the ID of the just being deployed Model, a "0" should be used,
    and the actual ID of the new DeployedModel will be filled in its place by
    this method. The traffic percentage values must add up to 100. If this
    field is empty, then the Endpoint's traffic_split is not updated.

    Messages:
      AdditionalProperty: An additional property for a TrafficSplitValue
        object.

    Fields:
      additionalProperties: Additional properties of type TrafficSplitValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TrafficSplitValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    deployedModel = _messages.MessageField('GoogleCloudAiplatformV1beta1DeployedModel', 1)
    trafficSplit = _messages.MessageField('TrafficSplitValue', 2)