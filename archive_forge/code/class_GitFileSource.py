from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitFileSource(_messages.Message):
    """GitFileSource describes a file within a (possibly remote) code
  repository.

  Enums:
    RepoTypeValueValuesEnum: See RepoType above.

  Fields:
    bitbucketServerConfig: The full resource name of the bitbucket server
      config. Format:
      `projects/{project}/locations/{location}/bitbucketServerConfigs/{id}`.
    githubEnterpriseConfig: The full resource name of the github enterprise
      config. Format:
      `projects/{project}/locations/{location}/githubEnterpriseConfigs/{id}`.
      `projects/{project}/githubEnterpriseConfigs/{id}`.
    path: The path of the file, with the repo root as the root of the path.
    repoType: See RepoType above.
    repository: The fully qualified resource name of the Repos API repository.
      Either URI or repository can be specified. If unspecified, the repo from
      which the trigger invocation originated is assumed to be the repo from
      which to read the specified path.
    revision: The branch, tag, arbitrary ref, or SHA version of the repo to
      use when resolving the filename (optional). This field respects the same
      syntax/resolution as described here: https://git-
      scm.com/docs/gitrevisions If unspecified, the revision from which the
      trigger invocation originated is assumed to be the revision from which
      to read the specified path.
    uri: The URI of the repo. Either uri or repository can be specified. If
      unspecified, the repo from which the trigger invocation originated is
      assumed to be the repo from which to read the specified path.
  """

    class RepoTypeValueValuesEnum(_messages.Enum):
        """See RepoType above.

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
    bitbucketServerConfig = _messages.StringField(1)
    githubEnterpriseConfig = _messages.StringField(2)
    path = _messages.StringField(3)
    repoType = _messages.EnumField('RepoTypeValueValuesEnum', 4)
    repository = _messages.StringField(5)
    revision = _messages.StringField(6)
    uri = _messages.StringField(7)