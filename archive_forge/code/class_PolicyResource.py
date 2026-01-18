import flask_restful
import http.client
from oslo_log import versionutils
from keystone.api._shared import json_home_relations
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
from keystone.policy import schema
from keystone.server import flask as ks_flask
class PolicyResource(ks_flask.ResourceBase):
    collection_key = 'policies'
    member_key = 'policy'

    def get(self, policy_id=None):
        if policy_id:
            return self._get_policy(policy_id)
        return self._list_policies()

    @versionutils.deprecated(as_of=versionutils.deprecated.QUEENS, what='identity:get_policy of the v3 Policy APIs')
    def _get_policy(self, policy_id):
        ENFORCER.enforce_call(action='identity:get_policy')
        ref = PROVIDERS.policy_api.get_policy(policy_id)
        return self.wrap_member(ref)

    @versionutils.deprecated(as_of=versionutils.deprecated.QUEENS, what='identity:list_policies of the v3 Policy APIs')
    def _list_policies(self):
        ENFORCER.enforce_call(action='identity:list_policies')
        filters = ['type']
        hints = self.build_driver_hints(filters)
        refs = PROVIDERS.policy_api.list_policies(hints=hints)
        return self.wrap_collection(refs, hints=hints)

    @versionutils.deprecated(as_of=versionutils.deprecated.QUEENS, what='identity:create_policy of the v3 Policy APIs')
    def post(self):
        ENFORCER.enforce_call(action='identity:create_policy')
        policy_body = self.request_body_json.get('policy', {})
        validation.lazy_validate(schema.policy_create, policy_body)
        policy = self._assign_unique_id(self._normalize_dict(policy_body))
        ref = PROVIDERS.policy_api.create_policy(policy['id'], policy, initiator=self.audit_initiator)
        return (self.wrap_member(ref), http.client.CREATED)

    @versionutils.deprecated(as_of=versionutils.deprecated.QUEENS, what='identity:update_policy of the v3 Policy APIs')
    def patch(self, policy_id):
        ENFORCER.enforce_call(action='identity:update_policy')
        policy_body = self.request_body_json.get('policy', {})
        validation.lazy_validate(schema.policy_update, policy_body)
        ref = PROVIDERS.policy_api.update_policy(policy_id, policy_body, initiator=self.audit_initiator)
        return self.wrap_member(ref)

    @versionutils.deprecated(as_of=versionutils.deprecated.QUEENS, what='identity:delete_policy of the v3 Policy APIs')
    def delete(self, policy_id):
        ENFORCER.enforce_call(action='identity:delete_policy')
        res = PROVIDERS.policy_api.delete_policy(policy_id, initiator=self.audit_initiator)
        return (res, http.client.NO_CONTENT)