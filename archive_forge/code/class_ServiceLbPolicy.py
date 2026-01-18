from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceLbPolicy(_messages.Message):
    """ServiceLbPolicy holds global load balancing and traffic distribution
  configuration that can be applied to a BackendService.

  Enums:
    LoadBalancingAlgorithmValueValuesEnum: Optional. The type of load
      balancing algorithm to be used. The default behavior is
      WATERFALL_BY_REGION.

  Messages:
    LabelsValue: Optional. Set of label tags associated with the
      ServiceLbPolicy resource.

  Fields:
    autoCapacityDrain: Optional. Configuration to automatically move traffic
      away for unhealthy IG/NEG for the associated Backend Service.
    createTime: Output only. The timestamp when this resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    failoverConfig: Optional. Configuration related to health based failover.
    labels: Optional. Set of label tags associated with the ServiceLbPolicy
      resource.
    loadBalancingAlgorithm: Optional. The type of load balancing algorithm to
      be used. The default behavior is WATERFALL_BY_REGION.
    name: Required. Name of the ServiceLbPolicy resource. It matches pattern `
      projects/{project}/locations/{location}/serviceLbPolicies/{service_lb_po
      licy_name}`.
    updateTime: Output only. The timestamp when this resource was last
      updated.
  """

    class LoadBalancingAlgorithmValueValuesEnum(_messages.Enum):
        """Optional. The type of load balancing algorithm to be used. The default
    behavior is WATERFALL_BY_REGION.

    Values:
      LOAD_BALANCING_ALGORITHM_UNSPECIFIED: The type of the loadbalancing
        algorithm is unspecified.
      SPRAY_TO_WORLD: Balance traffic across all backends across the world
        proportionally based on capacity.
      SPRAY_TO_REGION: Direct traffic to the nearest region with endpoints and
        capacity before spilling over to other regions and spread the traffic
        from each client to all the MIGs/NEGs in a region.
      WATERFALL_BY_REGION: Direct traffic to the nearest region with endpoints
        and capacity before spilling over to other regions. All MIGs/NEGs
        within a region are evenly loaded but each client might not spread the
        traffic to all the MIGs/NEGs in the region.
      WATERFALL_BY_ZONE: Attempt to keep traffic in a single zone closest to
        the client, before spilling over to other zones.
    """
        LOAD_BALANCING_ALGORITHM_UNSPECIFIED = 0
        SPRAY_TO_WORLD = 1
        SPRAY_TO_REGION = 2
        WATERFALL_BY_REGION = 3
        WATERFALL_BY_ZONE = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Set of label tags associated with the ServiceLbPolicy
    resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    autoCapacityDrain = _messages.MessageField('ServiceLbPolicyAutoCapacityDrain', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    failoverConfig = _messages.MessageField('ServiceLbPolicyFailoverConfig', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    loadBalancingAlgorithm = _messages.EnumField('LoadBalancingAlgorithmValueValuesEnum', 6)
    name = _messages.StringField(7)
    updateTime = _messages.StringField(8)