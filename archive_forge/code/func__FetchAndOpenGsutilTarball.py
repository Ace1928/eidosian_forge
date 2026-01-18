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
def _FetchAndOpenGsutilTarball(self, update_from_url_str):
    self.command_runner.RunNamedCommand('cp', [update_from_url_str, 'file://gsutil.tar.gz'], self.headers, self.debug, skip_update_check=True)
    tf = tarfile.open('gsutil.tar.gz')
    tf.errorlevel = 1
    return tf