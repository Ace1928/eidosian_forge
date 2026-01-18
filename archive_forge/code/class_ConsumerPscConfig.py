from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumerPscConfig(_messages.Message):
    """Allow the producer to specify which consumers can connect to it.

  Enums:
    StateValueValuesEnum: Output only. Overall state of PSC Connections
      management for this consumer psc config.

  Fields:
    disableGlobalAccess: This is used in PSC consumer ForwardingRule to
      control whether the PSC endpoint can be accessed from another region.
    network: The resource path of the consumer network where PSC connections
      are allowed to be created in. Note, this network does not need be in the
      ConsumerPscConfig.project in the case of SharedVPC. Example:
      projects/{projectNumOrId}/global/networks/{networkId}.
    project: The consumer project where PSC connections are allowed to be
      created in.
    state: Output only. Overall state of PSC Connections management for this
      consumer psc config.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Overall state of PSC Connections management for this
    consumer psc config.

    Values:
      STATE_UNSPECIFIED: Default state, when Connection Map is created
        initially.
      VALID: Set when policy and map configuration is valid, and their
        matching can lead to allowing creation of PSC Connections subject to
        other constraints like connections limit.
      CONNECTION_POLICY_MISSING: No Service Connection Policy found for this
        network and Service Class
      POLICY_LIMIT_REACHED: Service Connection Policy limit reached for this
        network and Service Class
    """
        STATE_UNSPECIFIED = 0
        VALID = 1
        CONNECTION_POLICY_MISSING = 2
        POLICY_LIMIT_REACHED = 3
    disableGlobalAccess = _messages.BooleanField(1)
    network = _messages.StringField(2)
    project = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)