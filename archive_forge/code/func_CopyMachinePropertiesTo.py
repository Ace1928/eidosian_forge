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
def CopyMachinePropertiesTo(self, other_state):
    """Copy this state's properties file to another state.

    This is primarily intended to be used to maintain the machine properties
    file during a schema-change-induced reinstall.

    Args:
      other_state: InstallationState, The installation state of the fresh
          Cloud SDK that needs the properties file mirrored in.
    """
    my_properties = os.path.join(self.sdk_root, config.Paths.CLOUDSDK_PROPERTIES_NAME)
    other_properties = os.path.join(other_state.sdk_root, config.Paths.CLOUDSDK_PROPERTIES_NAME)
    if not os.path.exists(my_properties):
        return
    shutil.copyfile(my_properties, other_properties)