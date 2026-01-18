from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.core import log
class VPCTroubleshooter(ssh_troubleshooter.SshTroubleshooter):
    """Check VPC setting."""
    project = None
    zone = None
    instance = None
    iap_tunnel_args = None

    def __init__(self, project, zone, instance, iap_tunnel_args):
        self.project = project
        self.zone = zone
        self.instance = instance
        self.iap_tunnel_args = iap_tunnel_args
        self.compute_client = apis.GetClientInstance(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.compute_message = apis.GetMessagesModule(_API_COMPUTE_CLIENT_NAME, _API_CLIENT_VERSION_V1)
        self.issues = {}

    def check_prerequisite(self):
        return

    def cleanup_resources(self):
        return

    def troubleshoot(self):
        log.status.Print('---- Checking VPC settings ----')
        self._CheckDefaultSSHPort()
        if self.iap_tunnel_args:
            self._CheckIAPTunneling()
        log.status.Print('VPC settings: {0} issue(s) found.\n'.format(len(self.issues)))
        for message in self.issues.values():
            log.status.Print(message)
        return

    def _CheckIAPTunneling(self):
        firewall_list = self._ListInstanceEffectiveFirewall()
        for firewall in firewall_list:
            if self._HasValidateIAPTunnelingRule(firewall):
                return
        self.issues['iap'] = IAP_MESSAGE

    def _CheckDefaultSSHPort(self):
        firewall_list = self._ListInstanceEffectiveFirewall()
        for firewall in firewall_list:
            if self._HasSSHProtocalAndPort(firewall):
                return
        self.issues['default_ssh_port'] = DEFAULT_SSH_PORT_MESSAGE

    def _ListInstanceEffectiveFirewall(self):
        req = self.compute_message.ComputeInstancesGetEffectiveFirewallsRequest(instance=self.instance.name, networkInterface='nic0', project=self.project.name, zone=self.zone)
        return self.compute_client.instances.GetEffectiveFirewalls(req).firewalls

    def _HasValidateIAPTunnelingRule(self, firewall):
        if firewall.direction != self.compute_message.Firewall.DirectionValueValuesEnum.INGRESS:
            return False
        if all((range != '35.235.240.0/20' for range in firewall.sourceRanges)):
            return False
        if not self._HasSSHProtocalAndPort(firewall):
            return False
        return True

    def _HasSSHProtocalAndPort(self, firewall):
        for allow_rule in firewall.allowed:
            if allow_rule.IPProtocol == 'tcp' and any((port == '22' for port in allow_rule.ports)):
                return True
        return False