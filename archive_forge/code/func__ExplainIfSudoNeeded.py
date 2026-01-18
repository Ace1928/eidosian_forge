from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import signal
import stat
import sys
import tarfile
import tempfile
import textwrap
from six.moves import input
import gslib
from gslib.command import Command
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.metrics import CheckAndMaybePromptForAnalyticsEnabling
from gslib.sig_handling import RegisterSignalHandler
from gslib.utils import system_util
from gslib.utils.boto_util import GetConfigFilePaths
from gslib.utils.boto_util import CERTIFICATE_VALIDATION_ENABLED
from gslib.utils.constants import RELEASE_NOTES_URL
from gslib.utils.text_util import CompareVersions
from gslib.utils.update_util import DisallowUpdateIfDataInGsutilDir
from gslib.utils.update_util import LookUpGsutilVersion
from gslib.utils.update_util import GsutilPubTarball
def _ExplainIfSudoNeeded(self, tf, dirs_to_remove, old_cwd):
    """Explains what to do if sudo needed to update gsutil software.

    Happens if gsutil was previously installed by a different user (typically if
    someone originally installed in a shared file system location, using sudo).

    Args:
      tf: Opened TarFile.
      dirs_to_remove: List of directories to remove.
      old_cwd: Path to the working directory we should chdir back to if sudo is
          needed. It's possible that we've chdir'd to a temp directory that's
          been deleted, which can cause odd behavior (e.g. OSErrors when opening
          the metrics subprocess). If this is not truthy, we won't attempt to
          chdir back to this value.

    Raises:
      CommandException: if errors encountered.
    """
    if system_util.IS_CYGWIN or system_util.IS_WINDOWS:
        return
    user_id = os.getuid()
    if os.stat(gslib.GSUTIL_DIR).st_uid == user_id:
        return
    config_file_list = GetConfigFilePaths()
    config_files = ' '.join(config_file_list)
    self._CleanUpUpdateCommand(tf, dirs_to_remove, old_cwd)
    chmod_cmds = []
    for config_file in config_file_list:
        mode = oct(stat.S_IMODE(os.stat(config_file)[stat.ST_MODE]))
        chmod_cmds.append('\n\tsudo chmod %s %s' % (mode, config_file))
    raise CommandException('\n'.join(textwrap.wrap('Since it was installed by a different user previously, you will need to update using the following commands. You will be prompted for your password, and the install will run as "root". If you\'re unsure what this means please ask your system administrator for help:')) + '\n\tsudo chmod 0644 %s\n\tsudo env BOTO_CONFIG="%s" %s update%s' % (config_files, config_files, self.gsutil_path, ' '.join(chmod_cmds)), informational=True)