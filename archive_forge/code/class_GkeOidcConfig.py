from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeOidcConfig(_messages.Message):
    """GkeOidcConfig is configuration for GKE OIDC which allows customers to
  use external OIDC providers with the K8S API

  Fields:
    enabled: Whether to enable the GKD OIDC component
  """
    enabled = _messages.BooleanField(1)