from keystoneclient import base
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import policies
def _act_on_policy_association_for_endpoint(self, policy, endpoint, action):
    if not (policy and endpoint):
        raise ValueError(_('policy and endpoint are required'))
    policy_id = base.getid(policy)
    endpoint_id = base.getid(endpoint)
    url = '/policies/%(policy_id)s/%(ext_name)s/endpoints/%(endpoint_id)s' % {'policy_id': policy_id, 'ext_name': self.OS_EP_POLICY_EXT, 'endpoint_id': endpoint_id}
    return action(url=url)