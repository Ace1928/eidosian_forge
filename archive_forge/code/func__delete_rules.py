from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
def _delete_rules(self, to_delete=None):
    try:
        sec = self.client().show_security_group(self.resource_id)['security_group']
    except Exception as ex:
        self.client_plugin().ignore_not_found(ex)
    else:
        for rule in sec['security_group_rules']:
            if to_delete is None or to_delete(rule):
                with self.client_plugin().ignore_not_found:
                    self.client().delete_security_group_rule(rule['id'])