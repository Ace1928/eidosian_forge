from openstack.cloud import _utils
from openstack.cloud import exc
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
def _normalize_secgroup(self, group):
    ret = utils.Munch()
    group = group.copy()
    self._remove_novaclient_artifacts(group)
    rules = self._normalize_secgroup_rules(group.pop('security_group_rules', group.pop('rules', [])))
    project_id = group.pop('tenant_id', '')
    project_id = group.pop('project_id', project_id)
    ret['location'] = self._get_current_location(project_id=project_id)
    ret['id'] = group.pop('id')
    ret['name'] = group.pop('name')
    ret['security_group_rules'] = rules
    ret['description'] = group.pop('description')
    ret['properties'] = group
    if self._use_neutron_secgroups():
        ret['stateful'] = group.pop('stateful', True)
    if not self.strict_mode:
        ret['tenant_id'] = project_id
        ret['project_id'] = project_id
        for key, val in ret['properties'].items():
            ret.setdefault(key, val)
    return ret