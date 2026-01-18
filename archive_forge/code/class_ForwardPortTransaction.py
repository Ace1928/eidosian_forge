from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class ForwardPortTransaction(FirewallTransaction):
    """
    ForwardPortTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=False, immediate=False):
        super(ForwardPortTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate)

    def get_enabled_immediate(self, port, proto, toport, toaddr, timeout):
        if self.fw_offline:
            dummy, fw_settings = self.get_fw_zone_settings()
            return fw_settings.queryForwardPort(port=port, protocol=proto, to_port=toport, to_addr=toaddr)
        return self.fw.queryForwardPort(zone=self.zone, port=port, protocol=proto, toport=toport, toaddr=toaddr)

    def get_enabled_permanent(self, port, proto, toport, toaddr, timeout):
        dummy, fw_settings = self.get_fw_zone_settings()
        return fw_settings.queryForwardPort(port=port, protocol=proto, to_port=toport, to_addr=toaddr)

    def set_enabled_immediate(self, port, proto, toport, toaddr, timeout):
        self.fw.addForwardPort(self.zone, port, proto, toport, toaddr, timeout)

    def set_enabled_permanent(self, port, proto, toport, toaddr, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.addForwardPort(port, proto, toport, toaddr)
        self.update_fw_settings(fw_zone, fw_settings)

    def set_disabled_immediate(self, port, proto, toport, toaddr, timeout):
        self.fw.removeForwardPort(self.zone, port, proto, toport, toaddr)

    def set_disabled_permanent(self, port, proto, toport, toaddr, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.removeForwardPort(port, proto, toport, toaddr)
        self.update_fw_settings(fw_zone, fw_settings)