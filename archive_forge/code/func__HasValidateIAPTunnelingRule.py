from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
def _HasValidateIAPTunnelingRule(self, firewall):
    if firewall.direction != self.compute_message.Firewall.DirectionValueValuesEnum.INGRESS:
        return False
    if all((range != '35.235.240.0/20' for range in firewall.sourceRanges)):
        return False
    if not self._HasSSHProtocalAndPort(firewall):
        return False
    return True