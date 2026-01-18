from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import os
import random
import re
import socket
import subprocess
import tempfile
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
def PrintEnvUnset(env):
    """Print OS specific unset commands for the given environment variables.

  Args:
    env: {str: str}, Dictionary of environment values, the value is ignored.
  """
    current_os = platforms.OperatingSystem.Current()
    export_command = 'unset {var}'
    if current_os is platforms.OperatingSystem.WINDOWS:
        export_command = 'set {var}='
    for var in six.iterkeys(env):
        log.Print(export_command.format(var=var))