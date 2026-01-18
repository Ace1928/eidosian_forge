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
def GenerateConfigs(self):
    """Generates all config files for the module.

    Returns:
      (bool) True if files were written.
    """
    all_config_files = []
    if not self.params.appinfo:
        all_config_files.append(self._GenerateAppYaml())
    if self.params.custom or self.params.deploy:
        all_config_files.append(self._GenerateDockerfile())
        all_config_files.append(self._GenerateDockerignore())
    created = [config_file.WriteTo(self.root, self.notify) for config_file in all_config_files]
    if not any(created):
        self.notify('All config files already exist. No files generated.')
    return any(created)