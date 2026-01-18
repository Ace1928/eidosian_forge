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
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
def _list_groups(self):
    """List groups.

        GET/HEAD /groups
        """
    filters = ['domain_id', 'name']
    target = None
    if self.oslo_context.domain_id:
        target = {'group': {'domain_id': self.oslo_context.domain_id}}
    ENFORCER.enforce_call(action='identity:list_groups', filters=filters, target_attr=target)
    hints = self.build_driver_hints(filters)
    domain = self._get_domain_id_for_list_request()
    refs = PROVIDERS.identity_api.list_groups(domain_scope=domain, hints=hints)
    if self.oslo_context.domain_id:
        filtered_refs = []
        for ref in refs:
            if ref['domain_id'] == target['group']['domain_id']:
                filtered_refs.append(ref)
        refs = filtered_refs
    return self.wrap_collection(refs, hints=hints)