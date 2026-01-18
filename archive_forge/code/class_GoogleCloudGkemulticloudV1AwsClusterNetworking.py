from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsClusterNetworking(_messages.Message):
    """ClusterNetworking defines cluster-wide networking configuration. Anthos
  clusters on AWS run on a single VPC. This includes control plane replicas
  and node pool nodes.

  Fields:
    perNodePoolSgRulesDisabled: Optional. Disable the per node pool subnet
      security group rules on the control plane security group. When set to
      true, you must also provide one or more security groups that ensure node
      pools are able to send requests to the control plane on TCP/443 and
      TCP/8132. Failure to do so may result in unavailable node pools.
    podAddressCidrBlocks: Required. All pods in the cluster are assigned an
      IPv4 address from these ranges. Only a single range is supported. This
      field cannot be changed after creation.
    serviceAddressCidrBlocks: Required. All services in the cluster are
      assigned an IPv4 address from these ranges. Only a single range is
      supported. This field cannot be changed after creation.
    vpcId: Required. The VPC associated with the cluster. All component
      clusters (i.e. control plane and node pools) run on a single VPC. This
      field cannot be changed after creation.
  """
    perNodePoolSgRulesDisabled = _messages.BooleanField(1)
    podAddressCidrBlocks = _messages.StringField(2, repeated=True)
    serviceAddressCidrBlocks = _messages.StringField(3, repeated=True)
    vpcId = _messages.StringField(4)