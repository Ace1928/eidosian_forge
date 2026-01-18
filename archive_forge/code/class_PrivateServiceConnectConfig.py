from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateServiceConnectConfig(_messages.Message):
    """Configuration for using Private Service Connect to establish
  connectivity between the Data Fusion consumer project and the corresponding
  tenant project.

  Fields:
    effectiveUnreachableCidrBlock: Output only. The CIDR block to which the
      CDF instance can't route traffic to in the consumer project VPC. The
      size of this block is /25. The format of this field is governed by RFC
      4632. Example: 240.0.0.0/25
    networkAttachment: Required. The reference to the network attachment used
      to establish private connectivity. It will be of the form
      projects/{project-id}/regions/{region}/networkAttachments/{network-
      attachment-id}.
    unreachableCidrBlock: Optional. Input only. The CIDR block to which the
      CDF instance can't route traffic to in the consumer project VPC. The
      size of this block should be at least /25. This range should not overlap
      with the primary address range of any subnetwork used by the network
      attachment. This range can be used for other purposes in the consumer
      VPC as long as there is no requirement for CDF to reach destinations
      using these addresses. If this value is not provided, the server chooses
      a non RFC 1918 address range. The format of this field is governed by
      RFC 4632. Example: 192.168.0.0/25
  """
    effectiveUnreachableCidrBlock = _messages.StringField(1)
    networkAttachment = _messages.StringField(2)
    unreachableCidrBlock = _messages.StringField(3)