from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessDeniedPageSettings(_messages.Message):
    """Custom content configuration for access denied page. IAP allows
  customers to define a custom URI to use as the error page when access is
  denied to users. If IAP prevents access to this page, the default IAP error
  page will be displayed instead.

  Fields:
    accessDeniedPageUri: The URI to be redirected to when access is denied.
    generateTroubleshootingUri: Whether to generate a troubleshooting URL on
      access denied events to this application.
    remediationTokenGenerationEnabled: Whether to generate remediation token
      on access denied events to this application.
  """
    accessDeniedPageUri = _messages.StringField(1)
    generateTroubleshootingUri = _messages.BooleanField(2)
    remediationTokenGenerationEnabled = _messages.BooleanField(3)