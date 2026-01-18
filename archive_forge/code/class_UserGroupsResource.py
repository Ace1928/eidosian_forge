import base64
import secrets
import uuid
import flask
import http.client
from oslo_serialization import jsonutils
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.application_credential import schema as app_cred_schema
from keystone.common import json_home
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import utils
from keystone.common import validation
import keystone.conf
from keystone import exception as ks_exception
from keystone.i18n import _
from keystone.identity import schema
from keystone import notifications
from keystone.server import flask as ks_flask
class UserGroupsResource(ks_flask.ResourceBase):
    collection_key = 'groups'
    member_key = 'group'
    get_member_from_driver = PROVIDERS.deferred_provider_lookup(api='identity_api', method='get_group')

    def get(self, user_id):
        """Get groups for a user.

        GET/HEAD /v3/users/{user_id}/groups
        """
        filters = ('name',)
        hints = self.build_driver_hints(filters)
        ENFORCER.enforce_call(action='identity:list_groups_for_user', build_target=_build_user_target_enforcement, filters=filters)
        refs = PROVIDERS.identity_api.list_groups_for_user(user_id=user_id, hints=hints)
        if self.oslo_context.domain_id:
            filtered_refs = []
            for ref in refs:
                if ref['domain_id'] == self.oslo_context.domain_id:
                    filtered_refs.append(ref)
            refs = filtered_refs
        return self.wrap_collection(refs, hints=hints)