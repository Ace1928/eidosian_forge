from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _ChooseEntrypoint(default_entrypoint, appinfo):
    """Prompt the user for an entrypoint.

  Args:
    default_entrypoint: (str) Default entrypoint determined from the app.
    appinfo: (apphosting.api.appinfo.AppInfoExternal or None) The parsed
      app.yaml file for the module if it exists.

  Returns:
    (str) The actual entrypoint to use.

  Raises:
    RubyConfigError: Unable to get entrypoint from the user.
  """
    if console_io.CanPrompt():
        if default_entrypoint:
            prompt = '\nPlease enter the command to run this Ruby app in production, or leave blank to accept the default:\n[{0}] '
            entrypoint = console_io.PromptResponse(prompt.format(default_entrypoint))
        else:
            entrypoint = console_io.PromptResponse('\nPlease enter the command to run this Ruby app in production: ')
        entrypoint = entrypoint.strip()
        if not entrypoint:
            if not default_entrypoint:
                raise RubyConfigError('Entrypoint command is required.')
            entrypoint = default_entrypoint
        if appinfo:
            msg = '\nTo avoid being asked for an entrypoint in the future, please add it to your app.yaml. e.g.\n  entrypoint: {0}'.format(entrypoint)
            log.status.Print(msg)
        return entrypoint
    else:
        msg = 'This appears to be a Ruby app. You\'ll need to provide the full command to run the app in production, but gcloud is not running interactively and cannot ask for the entrypoint{0}. Please either run gcloud interactively, or create an app.yaml with "runtime:ruby" and an "entrypoint" field.'.format(ext_runtime_adapter.GetNonInteractiveErrorMessage())
        raise RubyConfigError(msg)