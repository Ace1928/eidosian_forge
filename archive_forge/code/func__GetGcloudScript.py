from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import subprocess
import textwrap
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
import uritemplate
def _GetGcloudScript(full_path=False):
    """Get name of the gcloud script.

  Args:
    full_path: boolean, True if the gcloud full path should be used if free
      of spaces.

  Returns:
    str, command to use to execute gcloud

  Raises:
    GcloudIsNotInPath: if gcloud is not found in the path
  """
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        gcloud_ext = '.cmd'
    else:
        gcloud_ext = ''
    gcloud_name = 'gcloud'
    gcloud = files.FindExecutableOnPath(gcloud_name, pathext=[gcloud_ext])
    if not gcloud:
        raise GcloudIsNotInPath('Could not verify that gcloud is in the PATH. Please make sure the Cloud SDK bin folder is in PATH.')
    if full_path:
        if not re.match('[-a-zA-Z0-9_/]+$', gcloud):
            log.warning(textwrap.dedent('          You specified the option to use the full gcloud path in the git\n          credential.helper, but the path contains non alphanumberic characters\n          so the credential helper may not work correctly.'))
        return gcloud
    else:
        return gcloud_name + gcloud_ext