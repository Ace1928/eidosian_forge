import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _ParseSourceContext(remote_name, remote_url, source_revision):
    """Parses the URL into a source context blob, if the URL is a git or GCP repo.

  Args:
    remote_name: The name of the remote.
    remote_url: The remote URL to parse.
    source_revision: The current revision of the source directory.
  Returns:
    An ExtendedSourceContext suitable for JSON.
  """
    context = None
    match = re.match(_CLOUD_REPO_PATTERN, remote_url)
    if match:
        id_type = match.group('id_type')
        if id_type == 'id':
            raw_repo_id = match.group('project_or_repo_id')
            if not match.group('repo_name'):
                context = {'cloudRepo': {'repoId': {'uid': raw_repo_id}, 'revisionId': source_revision}}
        elif id_type == 'p':
            project_id = match.group('project_or_repo_id')
            repo_name = match.group('repo_name') or 'default'
            context = {'cloudRepo': {'repoId': {'projectRepoId': {'projectId': project_id, 'repoName': repo_name}}, 'revisionId': source_revision}}
    if not context:
        context = {'git': {'url': remote_url, 'revisionId': source_revision}}
    return ExtendContextDict(context, remote_name=remote_name)