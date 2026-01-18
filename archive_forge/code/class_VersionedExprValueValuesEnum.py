from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VersionedExprValueValuesEnum(_messages.Enum):
    """Preconfigured versioned expression. If this field is specified, config
    must also be specified. Available preconfigured expressions along with
    their requirements are: SRC_IPS_V1 - must specify the corresponding
    src_ip_range field in config.

    Values:
      FIREWALL: <no description>
      SRC_IPS_V1: Matches the source IP address of a request to the IP ranges
        supplied in config.
    """
    FIREWALL = 0
    SRC_IPS_V1 = 1