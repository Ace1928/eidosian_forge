import flask_restful
from keystone.api._shared import implied_roles as shared
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.server import flask as ks_flask
class RoleInferencesResource(flask_restful.Resource):

    def get(self):
        """List role inference rules.

        GET/HEAD /v3/role_inferences
        """
        ENFORCER.enforce_call(action='identity:list_role_inference_rules')
        refs = PROVIDERS.role_api.list_role_inference_rules()
        role_dict = {role_ref['id']: role_ref for role_ref in PROVIDERS.role_api.list_roles()}
        rules = dict()
        for ref in refs:
            implied_role_id = ref['implied_role_id']
            prior_role_id = ref['prior_role_id']
            implied = rules.get(prior_role_id, [])
            implied.append(shared.build_implied_role_response_data(role_dict[implied_role_id]))
            rules[prior_role_id] = implied
        inferences = []
        for prior_id, implied in rules.items():
            prior_response = shared.build_prior_role_response_data(prior_id, role_dict[prior_id]['name'])
            inferences.append({'prior_role': prior_response, 'implies': implied})
        results = {'role_inferences': inferences}
        return results