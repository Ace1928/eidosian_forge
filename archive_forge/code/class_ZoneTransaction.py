from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class ZoneTransaction(FirewallTransaction):
    """
    ZoneTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=True, immediate=False, enabled_values=None, disabled_values=None):
        super(ZoneTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate, enabled_values=enabled_values or ['present'], disabled_values=disabled_values or ['absent'])
        self.enabled_msg = 'Added zone %s' % self.zone
        self.disabled_msg = 'Removed zone %s' % self.zone
        self.tx_not_permanent_error_msg = "Zone operations must be permanent. Make sure you didn't set the 'permanent' flag to 'false' or the 'immediate' flag to 'true'."

    def get_enabled_immediate(self):
        self.module.fail_json(msg=self.tx_not_permanent_error_msg)

    def get_enabled_permanent(self):
        if self.fw_offline:
            zones = self.fw.config.get_zones()
            zone_names = [self.fw.config.get_zone(z).name for z in zones]
        else:
            zones = self.fw.config().listZones()
            zone_names = [self.fw.config().getZone(z).get_property('name') for z in zones]
        return self.zone in zone_names

    def set_enabled_immediate(self):
        self.module.fail_json(msg=self.tx_not_permanent_error_msg)

    def set_enabled_permanent(self):
        if self.fw_offline:
            self.fw.config.new_zone(self.zone, FirewallClientZoneSettings().settings)
        else:
            self.fw.config().addZone(self.zone, FirewallClientZoneSettings())

    def set_disabled_immediate(self):
        self.module.fail_json(msg=self.tx_not_permanent_error_msg)

    def set_disabled_permanent(self):
        if self.fw_offline:
            zone = self.fw.config.get_zone(self.zone)
            self.fw.config.remove_zone(zone)
        else:
            zone_obj = self.fw.config().getZoneByName(self.zone)
            zone_obj.remove()