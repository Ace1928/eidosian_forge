import copy
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.senlin import res_base
from heat.engine import translation
def add_bindings(self, bindings):
    for bd in bindings:
        bd['action'] = self.client().attach_policy_to_cluster(bd[self.BD_CLUSTER], self.resource_id, enabled=bd[self.BD_ENABLED])['action']
        bd['finished'] = False