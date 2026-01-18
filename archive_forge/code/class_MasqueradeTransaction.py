from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class MasqueradeTransaction(FirewallTransaction):
    """
    MasqueradeTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=False, immediate=False):
        super(MasqueradeTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate)
        self.enabled_msg = 'Added masquerade to zone %s' % self.zone
        self.disabled_msg = 'Removed masquerade from zone %s' % self.zone

    def get_enabled_immediate(self):
        if self.fw.queryMasquerade(self.zone) is True:
            return True
        else:
            return False

    def get_enabled_permanent(self):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        if fw_settings.getMasquerade() is True:
            return True
        else:
            return False

    def set_enabled_immediate(self):
        self.fw.addMasquerade(self.zone)

    def set_enabled_permanent(self):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.setMasquerade(True)
        self.update_fw_settings(fw_zone, fw_settings)

    def set_disabled_immediate(self):
        self.fw.removeMasquerade(self.zone)

    def set_disabled_permanent(self):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.setMasquerade(False)
        self.update_fw_settings(fw_zone, fw_settings)