from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementOciConfig(_messages.Message):
    """OCI repo configuration for a single cluster

  Fields:
    gcpServiceAccountEmail: The Google Cloud Service Account Email used for
      auth when secret_type is gcpServiceAccount.
    policyDir: The absolute path of the directory that contains the local
      resources. Default: the root directory of the image.
    secretType: Type of secret configured for access to the Git repo.
    syncRepo: The OCI image repository URL for the package to sync from. e.g.
      `LOCATION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/PACKAGE_NAME`.
    syncWaitSecs: Period in seconds between consecutive syncs. Default: 15.
  """
    gcpServiceAccountEmail = _messages.StringField(1)
    policyDir = _messages.StringField(2)
    secretType = _messages.StringField(3)
    syncRepo = _messages.StringField(4)
    syncWaitSecs = _messages.IntegerField(5)