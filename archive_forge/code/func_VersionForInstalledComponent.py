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
def VersionForInstalledComponent(component_id):
    """Gets the version string for the given installed component.

    This function is to be used to get component versions for metrics reporting.
    If it fails in any way or if the component_id is unknown, it will return
    None.  This prevents errors from surfacing when the version is needed
    strictly for reporting purposes.

    Args:
      component_id: str, The component id of the component you want the version
        for.

    Returns:
      str, The installed version of the component, or None if it is not
        installed or if an error occurs.
    """
    try:
        state = InstallationState.ForCurrent()
        return InstallationManifest(state._state_directory, component_id).VersionString()
    except:
        logging.debug('Failed to get installed version for component [%s]: [%s]', component_id, sys.exc_info())
    return None