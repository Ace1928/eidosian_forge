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
class GroupUsersResource(ks_flask.ResourceBase):

    def get(self, group_id):
        """Get list of users in group.

        GET/HEAD /groups/{group_id}/users
        """
        filters = ['domain_id', 'enabled', 'name', 'password_expires_at']
        target = None
        try:
            target = {'group': PROVIDERS.identity_api.get_group(group_id)}
        except exception.GroupNotFound:
            pass
        ENFORCER.enforce_call(action='identity:list_users_in_group', target_attr=target, filters=filters)
        hints = ks_flask.ResourceBase.build_driver_hints(filters)
        refs = PROVIDERS.identity_api.list_users_in_group(group_id, hints=hints)
        if self.oslo_context.domain_id:
            filtered_refs = []
            for ref in refs:
                if ref['domain_id'] == self.oslo_context.domain_id:
                    filtered_refs.append(ref)
            refs = filtered_refs
        return ks_flask.ResourceBase.wrap_collection(refs, hints=hints, collection_name='users')