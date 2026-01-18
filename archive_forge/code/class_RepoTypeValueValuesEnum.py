from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepoTypeValueValuesEnum(_messages.Enum):
    """See RepoType below.

    Values:
      UNKNOWN: The default, unknown repo type. Don't use it, instead use one
        of the other repo types.
      CLOUD_SOURCE_REPOSITORIES: A Google Cloud Source Repositories-hosted
        repo.
      GITHUB: A GitHub-hosted repo not necessarily on "github.com" (i.e.
        GitHub Enterprise).
      BITBUCKET_SERVER: A Bitbucket Server-hosted repo.
      GITLAB: A GitLab-hosted repo.
      BITBUCKET_CLOUD: A Bitbucket Cloud-hosted repo.
    """
    UNKNOWN = 0
    CLOUD_SOURCE_REPOSITORIES = 1
    GITHUB = 2
    BITBUCKET_SERVER = 3
    GITLAB = 4
    BITBUCKET_CLOUD = 5