from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementGitConfig(_messages.Message):
    """Git repo configuration for a single cluster.

  Fields:
    gcpServiceAccountEmail: The Google Cloud Service Account Email used for
      auth when secret_type is gcpServiceAccount.
    httpsProxy: URL for the HTTPS proxy to be used when communicating with the
      Git repo.
    policyDir: The path within the Git repository that represents the top
      level of the repo to sync. Default: the root directory of the
      repository.
    secretType: Type of secret configured for access to the Git repo. Must be
      one of ssh, cookiefile, gcenode, token, gcpserviceaccount or none. The
      validation of this is case-sensitive. Required.
    syncBranch: The branch of the repository to sync from. Default: master.
    syncRepo: The URL of the Git repository to use as the source of truth.
    syncRev: Git revision (tag or hash) to check out. Default HEAD.
    syncWaitSecs: Period in seconds between consecutive syncs. Default: 15.
  """
    gcpServiceAccountEmail = _messages.StringField(1)
    httpsProxy = _messages.StringField(2)
    policyDir = _messages.StringField(3)
    secretType = _messages.StringField(4)
    syncBranch = _messages.StringField(5)
    syncRepo = _messages.StringField(6)
    syncRev = _messages.StringField(7)
    syncWaitSecs = _messages.IntegerField(8)