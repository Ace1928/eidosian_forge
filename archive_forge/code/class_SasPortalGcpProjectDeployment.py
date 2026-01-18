from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalGcpProjectDeployment(_messages.Message):
    """Deployment associated with the GCP project. Includes whether SAS
  analytics has been enabled or not.

  Fields:
    deployment: Deployment associated with the GCP project.
    hasEnabledAnalytics: Whether SAS analytics has been enabled.
  """
    deployment = _messages.MessageField('SasPortalDeployment', 1)
    hasEnabledAnalytics = _messages.BooleanField(2)