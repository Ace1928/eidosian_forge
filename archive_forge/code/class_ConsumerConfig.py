from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumerConfig(_messages.Message):
    """Configuration information for a private service access connection.

  Fields:
    cloudsqlConfigs: Represents one or multiple Cloud SQL configurations.
    consumerExportCustomRoutes: Export custom routes flag value for peering
      from consumer to producer.
    consumerExportSubnetRoutesWithPublicIp: Export subnet routes with public
      ip flag value for peering from consumer to producer.
    consumerImportCustomRoutes: Import custom routes flag value for peering
      from consumer to producer.
    consumerImportSubnetRoutesWithPublicIp: Import subnet routes with public
      ip flag value for peering from consumer to producer.
    producerExportCustomRoutes: Export custom routes flag value for peering
      from producer to consumer.
    producerExportSubnetRoutesWithPublicIp: Export subnet routes with public
      ip flag value for peering from producer to consumer.
    producerImportCustomRoutes: Import custom routes flag value for peering
      from producer to consumer.
    producerImportSubnetRoutesWithPublicIp: Import subnet routes with public
      ip flag value for peering from producer to consumer.
    producerNetwork: Output only. The VPC host network that is used to host
      managed service instances. In the format,
      projects/{project}/global/networks/{network} where {project} is the
      project number e.g. '12345' and {network} is the network name.
    reservedRanges: Output only. The reserved ranges associated with this
      private service access connection.
    usedIpRanges: Output only. The IP ranges already in use by consumer or
      producer
    vpcScReferenceArchitectureEnabled: Output only. Indicates whether the VPC
      Service Controls reference architecture is configured for the producer
      VPC host network.
  """
    cloudsqlConfigs = _messages.MessageField('CloudSQLConfig', 1, repeated=True)
    consumerExportCustomRoutes = _messages.BooleanField(2)
    consumerExportSubnetRoutesWithPublicIp = _messages.BooleanField(3)
    consumerImportCustomRoutes = _messages.BooleanField(4)
    consumerImportSubnetRoutesWithPublicIp = _messages.BooleanField(5)
    producerExportCustomRoutes = _messages.BooleanField(6)
    producerExportSubnetRoutesWithPublicIp = _messages.BooleanField(7)
    producerImportCustomRoutes = _messages.BooleanField(8)
    producerImportSubnetRoutesWithPublicIp = _messages.BooleanField(9)
    producerNetwork = _messages.StringField(10)
    reservedRanges = _messages.MessageField('GoogleCloudServicenetworkingV1ConsumerConfigReservedRange', 11, repeated=True)
    usedIpRanges = _messages.StringField(12, repeated=True)
    vpcScReferenceArchitectureEnabled = _messages.BooleanField(13)