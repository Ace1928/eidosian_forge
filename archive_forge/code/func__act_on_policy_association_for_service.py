from keystoneclient import base
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import policies
def _act_on_policy_association_for_service(self, policy, service, action):
    if not (policy and service):
        raise ValueError(_('policy and service are required'))
    policy_id = base.getid(policy)
    service_id = base.getid(service)
    url = '/policies/%(policy_id)s/%(ext_name)s/services/%(service_id)s' % {'policy_id': policy_id, 'ext_name': self.OS_EP_POLICY_EXT, 'service_id': service_id}
    return action(url=url)