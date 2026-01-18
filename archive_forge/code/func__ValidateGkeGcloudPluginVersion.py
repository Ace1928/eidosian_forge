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
def _ValidateGkeGcloudPluginVersion(command):
    """Validate Gke Gcloud Plugin Command to be used.

  GDCE will depend on the newest available version, so warn customers if they
  have an older version installed.

  Args:
    command: Gke Gcloud Plugin Command to be used.
  """
    result = subprocess.run([command, '--help'], timeout=5, check=False, capture_output=True, text=True)
    if '--project string' not in result.stderr and '--project string' not in result.stdout:
        log.critical(GKE_GCLOUD_AUTH_PLUGIN_NOT_UP_TO_DATE)