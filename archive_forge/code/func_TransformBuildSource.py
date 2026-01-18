from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.container.fleet import client as hub_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
def TransformBuildSource(r, undefined=''):
    """Returns the formatted build source.

  Args:
    r: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.
  Returns:
    The formatted build source.
  """
    messages = core_apis.GetMessagesModule('cloudbuild', 'v1')
    b = apitools_encoding.DictToMessage(r, messages.Build)
    if b.source is None:
        return undefined
    storage_source = b.source.storageSource
    repo_source = b.source.repoSource
    if storage_source is not None:
        bucket = storage_source.bucket
        obj = storage_source.object
        if bucket is None or obj is None:
            return undefined
        return 'gs://{0}/{1}'.format(bucket, obj)
    if repo_source is not None:
        repo_name = repo_source.repoName or 'default'
        branch_name = repo_source.branchName
        if branch_name is not None:
            return '{0}@{1}'.format(repo_name, branch_name)
        tag_name = repo_source.tagName
        if tag_name is not None:
            return '{0}@{1}'.format(repo_name, tag_name)
        commit_sha = repo_source.commitSha
        if commit_sha is not None:
            return '{0}@{1}'.format(repo_name, commit_sha)
    return undefined