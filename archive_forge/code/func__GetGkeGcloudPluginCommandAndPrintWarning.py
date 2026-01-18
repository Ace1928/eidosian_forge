from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import os
import subprocess
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
def _GetGkeGcloudPluginCommandAndPrintWarning():
    """Gets Gke Gcloud Plugin Command to be used.

  Returns Gke Gcloud Plugin Command to be used. Also,
  prints warning if plugin is not present or doesn't work correctly.

  Returns:
    string, Gke Gcloud Plugin Command to be used.
  """
    bin_name = 'gke-gcloud-auth-plugin'
    if platforms.OperatingSystem.IsWindows():
        bin_name = 'gke-gcloud-auth-plugin.exe'
    command = bin_name
    try:
        subprocess.run([command, '--version'], timeout=5, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _ValidateGkeGcloudPluginVersion(command)
    except Exception:
        try:
            sdk_bin_path = config.Paths().sdk_bin_path
            if sdk_bin_path is None:
                log.critical(GKE_GCLOUD_AUTH_PLUGIN_NOT_FOUND)
            else:
                sdk_path_bin_name = os.path.join(sdk_bin_path, command)
                subprocess.run([sdk_path_bin_name, '--version'], timeout=5, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                command = sdk_path_bin_name
        except Exception:
            log.critical(GKE_GCLOUD_AUTH_PLUGIN_NOT_FOUND)
    return command