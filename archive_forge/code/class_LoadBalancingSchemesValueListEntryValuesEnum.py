from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoadBalancingSchemesValueListEntryValuesEnum(_messages.Enum):
    """LoadBalancingSchemesValueListEntryValuesEnum enum type.

    Values:
      EXTERNAL: Signifies that this will be used for classic Application Load
        Balancers.
      EXTERNAL_MANAGED: Signifies that this will be used for Envoy-based
        global external Application Load Balancers.
      LOAD_BALANCING_SCHEME_UNSPECIFIED: If unspecified, the validation will
        try to infer the scheme from the backend service resources this Url
        map references. If the inference is not possible, EXTERNAL will be
        used as the default type.
    """
    EXTERNAL = 0
    EXTERNAL_MANAGED = 1
    LOAD_BALANCING_SCHEME_UNSPECIFIED = 2