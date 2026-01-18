from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
@staticmethod
def ForCurrent():
    """Gets the installation state for the SDK that this code is running in.

    Returns:
      InstallationState, The state for this area.

    Raises:
      InvalidSDKRootError: If this code is not running under a valid SDK.
    """
    sdk_root = config.Paths().sdk_root
    if not sdk_root:
        raise InvalidSDKRootError()
    return InstallationState(os.path.realpath(sdk_root))