from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAzureClustersGetJwksRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAzureClustersGetJwksRequest object.

  Fields:
    azureCluster: Required. The AzureCluster, which owns the JsonWebKeys.
      Format: `projects//locations//azureClusters/`
  """
    azureCluster = _messages.StringField(1, required=True)