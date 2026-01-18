from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngressSettingsValueValuesEnum(_messages.Enum):
    """The ingress settings for the function, controlling what traffic can
    reach it.

    Values:
      INGRESS_SETTINGS_UNSPECIFIED: Unspecified.
      ALLOW_ALL: Allow HTTP traffic from public and private sources.
      ALLOW_INTERNAL_ONLY: Allow HTTP traffic from only private VPC sources.
      ALLOW_INTERNAL_AND_GCLB: Allow HTTP traffic from private VPC sources and
        through GCLB.
    """
    INGRESS_SETTINGS_UNSPECIFIED = 0
    ALLOW_ALL = 1
    ALLOW_INTERNAL_ONLY = 2
    ALLOW_INTERNAL_AND_GCLB = 3