from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
def _HasSSHProtocalAndPort(self, firewall):
    for allow_rule in firewall.allowed:
        if allow_rule.IPProtocol == 'tcp' and any((port == '22' for port in allow_rule.ports)):
            return True
    return False