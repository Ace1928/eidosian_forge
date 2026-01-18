import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetGitHeadRevision(source_directory):
    """Finds the current HEAD revision for the given source directory.

  Args:
    source_directory: The path to directory containing the source code.
  Returns:
    The HEAD revision of the current branch, or None if the command failed.
  """
    raw_output = _CallGit(source_directory, 'rev-parse', 'HEAD')
    return raw_output.strip() if raw_output else None