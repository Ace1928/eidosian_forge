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
def _execute_and_return_docker_command(args, project, creds_file_path):
    """Generates, executes, and returns agent install and run command."""
    full_docker_command = _get_docker_command(args, project, creds_file_path)
    completed_process = subprocess.run(full_docker_command, check=False)
    if completed_process.returncode != 0:
        log.status.Print('\nCould not execute Docker command. Trying with "sudo".')
        sudo_full_docker_command = ['sudo'] + full_docker_command
        sudo_completed_process = subprocess.run(sudo_full_docker_command, check=False)
        if sudo_completed_process.returncode != 0:
            raise OSError('Error executing Docker command:\n{}'.format(' '.join(full_docker_command)))
        executed_docker_command = sudo_full_docker_command
    else:
        executed_docker_command = full_docker_command
    _log_created_agent(executed_docker_command)
    return executed_docker_command