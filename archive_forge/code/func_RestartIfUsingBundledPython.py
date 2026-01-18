from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
def RestartIfUsingBundledPython(sdk_root, args=None, command=None):
    current_os = platforms.OperatingSystem.Current()
    if current_os is platforms.OperatingSystem.WINDOWS and _IsPythonBundled(sdk_root):
        if not console_io.CanPrompt():
            gcloud_cmd_path = os.path.realpath(os.path.join(config.Paths().sdk_bin_path or '', 'gcloud.cmd'))
            log.error('Cannot use bundled Python installation to update Google Cloud CLI in\nnon-interactive mode. Please run again in interactive mode.\n\n\n\nIf you really want to run in non-interactive mode, please run the\nfollowing command before re-running this one:\n\n\n\n  FOR /F "delims=" %i in ( \'""{0}"" components copy-bundled-python\'\n  ) DO (\n    SET CLOUDSDK_PYTHON=%i\n  )\n\n(Substitute `%%i` for `%i` if in a .bat script.)'.format(gcloud_cmd_path))
            sys.exit(1)
        RestartCommand(args=args, command=command, python=CopyPython(), block=False)
        sys.exit(0)