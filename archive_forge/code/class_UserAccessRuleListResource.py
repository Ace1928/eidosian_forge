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
class UserAccessRuleListResource(ks_flask.ResourceBase):
    collection_key = 'access_rules'
    member_key = 'access_rule'

    def get(self, user_id):
        """List access rules for user.

        GET/HEAD /v3/users/{user_id}/access_rules
        """
        filters = ('service', 'path', 'method')
        ENFORCER.enforce_call(action='identity:list_access_rules', filters=filters, build_target=_build_user_target_enforcement)
        app_cred_api = PROVIDERS.application_credential_api
        hints = self.build_driver_hints(filters)
        refs = app_cred_api.list_access_rules_for_user(user_id, hints=hints)
        hints = self.build_driver_hints(filters)
        return self.wrap_collection(refs, hints=hints)