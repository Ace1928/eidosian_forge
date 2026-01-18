from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class SourceTransaction(FirewallTransaction):
    """
    SourceTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=False, immediate=False):
        super(SourceTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate)
        self.enabled_msg = 'Added %s to zone %s' % (self.action_args[0], self.zone)
        self.disabled_msg = 'Removed %s from zone %s' % (self.action_args[0], self.zone)

    def get_enabled_immediate(self, source):
        if source in self.fw.getSources(self.zone):
            return True
        else:
            return False

    def get_enabled_permanent(self, source):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        if source in fw_settings.getSources():
            return True
        else:
            return False

    def set_enabled_immediate(self, source):
        self.fw.addSource(self.zone, source)

    def set_enabled_permanent(self, source):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.addSource(source)
        self.update_fw_settings(fw_zone, fw_settings)

    def set_disabled_immediate(self, source):
        self.fw.removeSource(self.zone, source)

    def set_disabled_permanent(self, source):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.removeSource(source)
        self.update_fw_settings(fw_zone, fw_settings)