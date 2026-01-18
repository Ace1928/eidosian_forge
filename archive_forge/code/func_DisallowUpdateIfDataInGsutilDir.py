from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import re
import textwrap
import sys
import gslib
from gslib.utils.system_util import IS_OSX
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
from gslib.utils.constants import GSUTIL_PUB_TARBALL
from gslib.utils.constants import GSUTIL_PUB_TARBALL_PY2
def DisallowUpdateIfDataInGsutilDir(directory=gslib.GSUTIL_DIR):
    """Disallows the update command if files not in the gsutil distro are found.

  This prevents users from losing data if they are in the habit of running
  gsutil from the gsutil directory and leaving data in that directory.

  This will also detect someone attempting to run gsutil update from a git
  repo, since the top-level directory will contain git files and dirs (like
  .git) that are not distributed with gsutil.

  Args:
    directory: (str) The directory to use this functionality on.

  Raises:
    CommandException: if files other than those distributed with gsutil found.
  """
    manifest_lines = ['MANIFEST.in', 'third_party']
    try:
        with open(os.path.join(directory, 'MANIFEST.in'), 'r') as fp:
            for line in fp:
                if line.startswith('include '):
                    manifest_lines.append(line.split()[-1])
                elif re.match('recursive-include \\w+ \\*', line):
                    manifest_lines.append(line.split()[1])
    except IOError:
        logging.getLogger().warn('MANIFEST.in not found in %s.\nSkipping user data check.\n', directory)
        return
    addl_excludes = ('.coverage', '.DS_Store', '.github', '.style.yapf', '.yapfignore', '__pycache__', '.github')
    for filename in os.listdir(directory):
        if filename.endswith('.pyc') or filename in addl_excludes:
            continue
        if filename not in manifest_lines:
            raise CommandException('\n'.join(textwrap.wrap('A file (%s) that is not distributed with gsutil was found in the gsutil directory. The update command cannot run with user data in the gsutil directory.' % os.path.join(gslib.GSUTIL_DIR, filename))))