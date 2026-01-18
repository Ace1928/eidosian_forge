import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def CalculateExtendedSourceContexts(source_directory):
    """Generate extended source contexts for a directory.

  Scans the remotes and revision of the git repository at source_directory,
  returning one or more ExtendedSourceContext-compatible dictionaries describing
  the repositories.

  Currently, this function will return only the Google-hosted repository
  associated with the directory, if one exists.

  Args:
    source_directory: The path to directory containing the source code.
  Returns:
    One or more ExtendedSourceContext-compatible dictionaries describing
    the remote repository or repositories associated with the given directory.
  Raises:
    GenerateSourceContextError: if source context could not be generated.
  """
    remote_urls = _GetGitRemoteUrls(source_directory)
    if not remote_urls:
        raise GenerateSourceContextError('Could not list remote URLs from source directory: %s' % source_directory)
    source_revision = _GetGitHeadRevision(source_directory)
    if not source_revision:
        raise GenerateSourceContextError('Could not find HEAD revision from the source directory: %s' % source_directory)
    source_contexts = []
    for remote_name, remote_url in remote_urls.items():
        source_context = _ParseSourceContext(remote_name, remote_url, source_revision)
        if source_context and source_context not in source_contexts:
            source_contexts.append(source_context)
    if not source_contexts:
        raise GenerateSourceContextError('Could not find any repository in the remote URLs for source directory: %s' % source_directory)
    return source_contexts