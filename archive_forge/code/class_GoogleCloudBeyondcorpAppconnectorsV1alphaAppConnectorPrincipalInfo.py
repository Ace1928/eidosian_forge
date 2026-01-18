from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnectorPrincipalInfo(_messages.Message):
    """PrincipalInfo represents an Identity oneof.

  Fields:
    serviceAccount: A GCP service account.
  """
    serviceAccount = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1alphaAppConnectorPrincipalInfoServiceAccount', 1)