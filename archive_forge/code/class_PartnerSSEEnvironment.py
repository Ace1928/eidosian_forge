from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartnerSSEEnvironment(_messages.Message):
    """Message describing PartnerSSEEnvironment object.

  Enums:
    SseServiceValueValuesEnum: Immutable. Only SYMANTEC_CLOUD_SWG uses
      PartnerSSEEnvironment today.

  Messages:
    LabelsValue: Optional. Labels as key value pair

  Fields:
    createTime: Output only. [Output only] Create time stamp
    deleteTime: Output only. [Output only] Delete time stamp
    labels: Optional. Labels as key value pair
    name: Identifier. Name of the Partner SSE Environment. Partner SSE
      Environment is global so the name should be unique per project. Partners
      should use the name "default" for the environment that want customers to
      use. See google.aip.dev/122 for resource naming.
    partnerNetwork: Required. Partner-owned network in the partner project
      created for this environment. Supports all user traffic and peers to
      sse_network.
    sseNetwork: Output only. Google-owned VPC in the SSE project created for
      this environment. Supports all user traffic and peers to partner_vpc.
    sseNetworkingRanges: Required. CIDR ranges reserved for Google's use.
      Should be at least a /20.
    sseProject: Output only. Google-owned project created for this
      environment.
    sseService: Immutable. Only SYMANTEC_CLOUD_SWG uses PartnerSSEEnvironment
      today.
    symantecOptions: Optional. Required iff sse_service is SYMANTEC_CLOUD_SWG.
    updateTime: Output only. [Output only] Update time stamp
  """

    class SseServiceValueValuesEnum(_messages.Enum):
        """Immutable. Only SYMANTEC_CLOUD_SWG uses PartnerSSEEnvironment today.

    Values:
      SSE_SERVICE_UNSPECIFIED: The default value. This value is used if the
        state is omitted.
      PALO_ALTO_PRISMA_ACCESS: [Palo Alto Networks Prisma
        Access](https://www.paloaltonetworks.com/sase/access).
      SYMANTEC_CLOUD_SWG: Symantec Cloud SWG is not fully supported yet - see
        b/323856877.
    """
        SSE_SERVICE_UNSPECIFIED = 0
        PALO_ALTO_PRISMA_ACCESS = 1
        SYMANTEC_CLOUD_SWG = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pair

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
    createTime = _messages.StringField(1)
    deleteTime = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    name = _messages.StringField(4)
    partnerNetwork = _messages.StringField(5)
    sseNetwork = _messages.StringField(6)
    sseNetworkingRanges = _messages.StringField(7, repeated=True)
    sseProject = _messages.StringField(8)
    sseService = _messages.EnumField('SseServiceValueValuesEnum', 9)
    symantecOptions = _messages.MessageField('PartnerSSEEnvironmentSymantecEnvironmentOptions', 10)
    updateTime = _messages.StringField(11)