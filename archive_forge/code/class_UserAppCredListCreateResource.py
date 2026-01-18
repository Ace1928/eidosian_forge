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
class UserAppCredListCreateResource(ks_flask.ResourceBase):
    collection_key = 'application_credentials'
    member_key = 'application_credential'
    _public_parameters = frozenset(['id', 'name', 'description', 'expires_at', 'project_id', 'roles', 'secret', 'links', 'unrestricted', 'access_rules'])

    @staticmethod
    def _generate_secret():
        length = 64
        secret = secrets.token_bytes(length)
        secret = base64.urlsafe_b64encode(secret)
        secret = secret.rstrip(b'=')
        secret = secret.decode('utf-8')
        return secret

    @staticmethod
    def _normalize_role_list(app_cred_roles):
        roles = []
        for role in app_cred_roles:
            if role.get('id'):
                roles.append(role)
            else:
                roles.append(PROVIDERS.role_api.get_unique_role_by_name(role['name']))
        return roles

    def _get_roles(self, app_cred_data, token):
        if app_cred_data.get('roles'):
            roles = self._normalize_role_list(app_cred_data['roles'])
            token_roles = [r['id'] for r in token.roles]
            for role in roles:
                if role['id'] not in token_roles:
                    detail = _('Cannot create an application credential with unassigned role')
                    raise ks_exception.ApplicationCredentialValidationError(detail=detail)
        else:
            roles = token.roles
        return roles

    def get(self, user_id):
        """List application credentials for user.

        GET/HEAD /v3/users/{user_id}/application_credentials
        """
        filters = ('name',)
        ENFORCER.enforce_call(action='identity:list_application_credentials', filters=filters)
        app_cred_api = PROVIDERS.application_credential_api
        hints = self.build_driver_hints(filters)
        refs = app_cred_api.list_application_credentials(user_id, hints=hints)
        return self.wrap_collection(refs, hints=hints)

    def post(self, user_id):
        """Create application credential.

        POST /v3/users/{user_id}/application_credentials
        """
        ENFORCER.enforce_call(action='identity:create_application_credential')
        app_cred_data = self.request_body_json.get('application_credential', {})
        validation.lazy_validate(app_cred_schema.application_credential_create, app_cred_data)
        token = self.auth_context['token']
        _check_unrestricted_application_credential(token)
        if self.oslo_context.user_id != user_id:
            action = _('Cannot create an application credential for another user.')
            raise ks_exception.ForbiddenAction(action=action)
        project_id = self.oslo_context.project_id
        app_cred_data = self._assign_unique_id(app_cred_data)
        if not app_cred_data.get('secret'):
            app_cred_data['secret'] = self._generate_secret()
        app_cred_data['user_id'] = user_id
        app_cred_data['project_id'] = project_id
        app_cred_data['roles'] = self._get_roles(app_cred_data, token)
        if app_cred_data.get('expires_at'):
            app_cred_data['expires_at'] = utils.parse_expiration_date(app_cred_data['expires_at'])
        if app_cred_data.get('access_rules'):
            for access_rule in app_cred_data['access_rules']:
                if 'id' not in access_rule:
                    access_rule['id'] = uuid.uuid4().hex
        app_cred_data = self._normalize_dict(app_cred_data)
        app_cred_api = PROVIDERS.application_credential_api
        try:
            ref = app_cred_api.create_application_credential(app_cred_data, initiator=self.audit_initiator)
        except ks_exception.RoleAssignmentNotFound as e:
            raise ks_exception.ApplicationCredentialValidationError(detail=str(e))
        return (self.wrap_member(ref), http.client.CREATED)