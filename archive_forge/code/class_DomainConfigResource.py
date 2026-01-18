import flask
import flask_restful
import functools
import http.client
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.resource import schema
from keystone.server import flask as ks_flask
class DomainConfigResource(DomainConfigBase):
    """Provides config routing functionality.

    This class leans on DomainConfigBase to provide the following APIs:

    GET/HEAD /v3/domains/{domain_id}/config
    PATCH /v3/domains/{domain_id}/config
    DELETE /v3/domains/{domain_id}/config
    """

    def put(self, domain_id):
        """Create domain config.

        PUT /v3/domains/{domain_id}/config
        """
        ENFORCER.enforce_call(action='identity:create_domain_config')
        PROVIDERS.resource_api.get_domain(domain_id)
        config = self.request_body_json.get('config', {})
        original_config = PROVIDERS.domain_config_api.get_config_with_sensitive_info(domain_id)
        ref = PROVIDERS.domain_config_api.create_config(domain_id, config)
        if original_config:
            return {self.member_key: ref}
        else:
            return ({self.member_key: ref}, http.client.CREATED)