from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import subprocess
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
import uritemplate
def CheckGitVersion(version_lower_bound=None):
    """Returns true when version of git is >= min_version.

  Args:
    version_lower_bound: (int,int,int), The lowest allowed version, or None to
      just check for the presence of git.

  Returns:
    True if version >= min_version.

  Raises:
    GitVersionException: if `git` was found, but the version is incorrect.
    InvalidGitException: if `git` was found, but the output of `git version` is
      not as expected.
    NoGitException: if `git` was not found.
  """
    try:
        cur_version = encoding.Decode(subprocess.check_output(['git', 'version']))
        if not cur_version:
            raise InvalidGitException('The git version string is empty.')
        if not cur_version.startswith('git version '):
            raise InvalidGitException('The git version string must start with git version .')
        match = re.search('(\\d+)\\.(\\d+)\\.(\\d+)', cur_version)
        if not match:
            raise InvalidGitException('The git version string must contain a version number.')
        current_version = tuple([int(item) for item in match.group(1, 2, 3)])
        if version_lower_bound and current_version < version_lower_bound:
            min_version = '.'.join((six.text_type(i) for i in version_lower_bound))
            raise GitVersionException('Your git version {cur_version} is older than the minimum version {min_version}. Please install a newer version of git.', cur_version=cur_version, min_version=min_version)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise NoGitException()
        raise
    return True