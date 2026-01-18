from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnPremDomainSIDDetails(_messages.Message):
    """OnPremDomainDetails is the message which contains details of on-prem
  domain which is trusted and needs to be migrated.

  Enums:
    SidFilteringStateValueValuesEnum: Current SID filtering state.

  Fields:
    name: FQDN of the on-prem domain being migrated.
    sidFilteringState: Current SID filtering state.
  """

    class SidFilteringStateValueValuesEnum(_messages.Enum):
        """Current SID filtering state.

    Values:
      SID_FILTERING_STATE_UNSPECIFIED: SID Filtering is in unspecified state.
      ENABLED: SID Filtering is Enabled.
      DISABLED: SID Filtering is Disabled.
    """
        SID_FILTERING_STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
    name = _messages.StringField(1)
    sidFilteringState = _messages.EnumField('SidFilteringStateValueValuesEnum', 2)