from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthValueValuesEnum(_messages.Enum):
    """The specified Istio auth mode, either none, or mutual TLS.

    Values:
      AUTH_NONE: auth not enabled
      AUTH_MUTUAL_TLS: auth mutual TLS enabled
    """
    AUTH_NONE = 0
    AUTH_MUTUAL_TLS = 1