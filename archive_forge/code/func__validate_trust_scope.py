from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _validate_trust_scope(self):
    trust_roles = []
    if self.trust_id:
        refs = [{'role_id': role['id']} for role in self.trust['roles']]
        effective_trust_roles = PROVIDERS.assignment_api.add_implied_roles(refs)
        effective_trust_role_ids = set([r['role_id'] for r in effective_trust_roles])
        current_effective_trustor_roles = PROVIDERS.assignment_api.get_roles_for_trustor_and_project(self.trustor['id'], self.trust.get('project_id'))
        for trust_role_id in effective_trust_role_ids:
            if trust_role_id in current_effective_trustor_roles:
                role = PROVIDERS.role_api.get_role(trust_role_id)
                if role['domain_id'] is None:
                    trust_roles.append(role)
            else:
                raise exception.Forbidden(_('Trustee has no delegated roles.'))