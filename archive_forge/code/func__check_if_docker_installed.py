from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
import socket
import subprocess
import sys
from googlecloudsdk.api_lib.transfer import agent_pools_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import platforms
from oauth2client import client as oauth2_client
def _check_if_docker_installed():
    """Checks for 'docker' in system PATH."""
    if not shutil.which('docker'):
        log.error('[2/3] Docker not found')
        if platforms.OperatingSystem.Current() == platforms.OperatingSystem.LINUX:
            error_format = DOCKER_NOT_FOUND_HELP_TEXT_LINUX_FORMAT
        else:
            error_format = DOCKER_NOT_FOUND_HELP_TEXT_NON_LINUX_FORMAT
        raise OSError(error_format.format(executed_command=_get_executed_command()))