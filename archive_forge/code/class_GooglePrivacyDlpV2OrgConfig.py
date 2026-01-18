from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2OrgConfig(_messages.Message):
    """Project and scan location information. Only set when the parent is an
  org.

  Fields:
    location: The data to scan: folder, org, or project
    projectId: The project that will run the scan. The DLP service account
      that exists within this project must have access to all resources that
      are profiled, and the Cloud DLP API must be enabled.
  """
    location = _messages.MessageField('GooglePrivacyDlpV2DiscoveryStartingLocation', 1)
    projectId = _messages.StringField(2)