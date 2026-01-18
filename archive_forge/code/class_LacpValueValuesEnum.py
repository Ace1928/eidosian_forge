from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LacpValueValuesEnum(_messages.Enum):
    """[Output Only] Link Aggregation Control Protocol (LACP) constraints,
    which can take one of the following values: LACP_SUPPORTED,
    LACP_UNSUPPORTED

    Values:
      LACP_SUPPORTED: LACP_SUPPORTED: LACP is supported, and enabled by
        default on the Cross-Cloud Interconnect.
      LACP_UNSUPPORTED: LACP_UNSUPPORTED: LACP is not supported and is not be
        enabled on this port. GetDiagnostics shows bundleAggregationType as
        "static". GCP does not support LAGs without LACP, so
        requestedLinkCount must be 1.
    """
    LACP_SUPPORTED = 0
    LACP_UNSUPPORTED = 1