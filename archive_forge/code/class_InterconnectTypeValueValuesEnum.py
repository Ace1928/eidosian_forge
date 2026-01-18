from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectTypeValueValuesEnum(_messages.Enum):
    """Type of interconnect, which can take one of the following values: -
    PARTNER: A partner-managed interconnection shared between customers though
    a partner. - DEDICATED: A dedicated physical interconnection with the
    customer. Note that a value IT_PRIVATE has been deprecated in favor of
    DEDICATED.

    Values:
      DEDICATED: A dedicated physical interconnection with the customer.
      IT_PRIVATE: [Deprecated] A private, physical interconnection with the
        customer.
      PARTNER: A partner-managed interconnection shared between customers via
        partner.
    """
    DEDICATED = 0
    IT_PRIVATE = 1
    PARTNER = 2