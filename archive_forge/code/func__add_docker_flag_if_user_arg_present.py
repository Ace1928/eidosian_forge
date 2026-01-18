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
def _add_docker_flag_if_user_arg_present(user_args, docker_args):
    """Adds user flags values directly directly to docker command."""
    for user_arg, docker_flag in _ADD_IF_PRESENT_PAIRS:
        user_value = getattr(user_args, user_arg, None)
        if user_value is not None:
            docker_args.append('{}={}'.format(docker_flag, user_value))