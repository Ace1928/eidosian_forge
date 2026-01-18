from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
def _IsOsLoginEnabled(self):
    """Check whether OS Login is enabled on the VM.

    Returns:
      boolean, indicates whether OS Login is enabled.
    """
    oslogin_enabled = ssh.FeatureEnabledInMetadata(self.instance, self.project, ssh.OSLOGIN_ENABLE_METADATA_KEY)
    return bool(oslogin_enabled)