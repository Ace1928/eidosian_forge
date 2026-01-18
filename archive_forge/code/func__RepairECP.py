from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import enum
import json
import os
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import _mtls_helper
from googlecloudsdk.command_lib.auth import enterprise_certificate_config
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _RepairECP(cert_config_file_path):
    """Install ECP and update the ecp config to include the new binaries.

  Args:
    cert_config_file_path: The filepath of the active certificate config.

  See go/gcloud-ecp-repair.
  """
    properties.VALUES.context_aware.use_client_certificate.Set(False)
    from googlecloudsdk.core.updater import update_manager
    platform = _GetPlatform()
    updater = update_manager.UpdateManager(sdk_root=None, url=None, platform_filter=platform)
    already_installed = updater.EnsureInstalledAndRestart(['enterprise-certificate-proxy'], 'Device appears to be enrolled in Certificate Base Access but is missing criticial components. Installing enterprise-certificate-proxy and restarting gcloud.')
    if already_installed:
        enterprise_certificate_config.update_config(enterprise_certificate_config.platform_to_config(platform), output_file=cert_config_file_path)
        properties.VALUES.context_aware.use_client_certificate.Set(True)