from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepositoryTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of the SCM vendor the repository points to.

    Values:
      REPOSITORY_TYPE_UNSPECIFIED: If unspecified, RepositoryType defaults to
        GITHUB.
      GITHUB: The SCM repo is GITHUB.
      GITHUB_ENTERPRISE: The SCM repo is GITHUB Enterprise.
      GITLAB_ENTERPRISE: The SCM repo is GITLAB Enterprise.
      BITBUCKET_DATA_CENTER: The SCM repo is BITBUCKET Data Center.
      BITBUCKET_CLOUD: The SCM repo is BITBUCKET Cloud.
    """
    REPOSITORY_TYPE_UNSPECIFIED = 0
    GITHUB = 1
    GITHUB_ENTERPRISE = 2
    GITLAB_ENTERPRISE = 3
    BITBUCKET_DATA_CENTER = 4
    BITBUCKET_CLOUD = 5