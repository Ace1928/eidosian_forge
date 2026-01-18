from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def ExpandLocalDirAndVersion(directory):
    """Expand HOME relative (~) directory with optional git_ref.

  Args:
      directory: str, directory path in the format PATH[/][@git_ref].

  Returns:
      str, expanded full directory path with git_ref (if provided)
  """
    path = directory.split('@') if directory else ''
    full_dir = files.ExpandHomeDir(path[0])
    if len(path) == 2:
        full_dir += '@' + path[1]
    return full_dir