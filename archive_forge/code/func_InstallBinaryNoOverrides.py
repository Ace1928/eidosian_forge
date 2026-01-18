from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def InstallBinaryNoOverrides(binary_name, prompt):
    """Helper method for installing binary dependencies within command execs."""
    console_io.PromptContinue(message='Pausing command execution:', prompt_string=prompt, cancel_on_no=True, cancel_string='Aborting component install for {} and command execution.'.format(binary_name))
    platform = platforms.Platform.Current()
    update_manager_client = update_manager.UpdateManager(platform_filter=platform)
    update_manager_client.Install([binary_name])
    path_executable = files.FindExecutableOnPath(binary_name)
    if path_executable:
        return path_executable
    raise MissingExecutableException(binary_name, '{} binary not installed'.format(binary_name))