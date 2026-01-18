from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AzureServicesAuthentication(_messages.Message):
    """Authentication configuration for the management of Azure resources.

  Fields:
    applicationId: Required. The Azure Active Directory Application ID.
    tenantId: Required. The Azure Active Directory Tenant ID.
  """
    applicationId = _messages.StringField(1)
    tenantId = _messages.StringField(2)