from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SSERealm(_messages.Message):
    """Message describing SSERealm object

  Enums:
    SseServiceValueValuesEnum: Immutable. SSE service provider
    StateValueValuesEnum: Output only. [Output only] State of the realm

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    labels: Optional. Labels as key value pairs
    name: Immutable. name of resource. It matches pattern
      `projects/{project}/locations/global/sseRealms/{sseRealm}`
    pairingKey: Output only. [Output only] Key to be shared with SSE service
      provider to establish global handshake
    partnerSseEnvironment: Optional. Full URI of environment that this Realm
      is using. Only used in Symantec Realms today.
    sseService: Immutable. SSE service provider
    state: Output only. [Output only] State of the realm
    symantecOptions: Optional. Required only if using SYMANTEC_CLOUD_SWG.
    updateTime: Output only. [Output only] Update time stamp
  """

    class SseServiceValueValuesEnum(_messages.Enum):
        """Immutable. SSE service provider

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

    class StateValueValuesEnum(_messages.Enum):
        """Output only. [Output only] State of the realm

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      ATTACHED: This SSERealm is attached to a PartnerSSERealm, used only for
        Prisma Access.
      UNATTACHED: This SSERealm is not attached to a PartnerSSERealm, used
        only for Prisma Access.
      KEY_EXPIRED: This SSERealm is not attached to a PartnerSSERealm, and its
        pairing key has expired and needs key regeneration, used only for
        Prisma Access.
      KEY_VALIDATION_PENDING: API key is pending validation for Symantec.
      KEY_VALIDATED: API key validation succeeded for Symantec, and customers
        can proceed to further steps.
      KEY_INVALID: API key validation failed for Symantec, please use a new
        API key.
    """
        STATE_UNSPECIFIED = 0
        ATTACHED = 1
        UNATTACHED = 2
        KEY_EXPIRED = 3
        KEY_VALIDATION_PENDING = 4
        KEY_VALIDATED = 5
        KEY_INVALID = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

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
    labels = _messages.MessageField('LabelsValue', 2)
    name = _messages.StringField(3)
    pairingKey = _messages.MessageField('SSERealmPairingKey', 4)
    partnerSseEnvironment = _messages.StringField(5)
    sseService = _messages.EnumField('SseServiceValueValuesEnum', 6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    symantecOptions = _messages.MessageField('SSERealmSSERealmSymantecOptions', 8)
    updateTime = _messages.StringField(9)