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
class UserChangePasswordResource(ks_flask.ResourceBase):

    @ks_flask.unenforced_api
    def get(self, user_id):
        raise exceptions.MethodNotAllowed(valid_methods=['POST'])

    @ks_flask.unenforced_api
    def post(self, user_id):
        user_data = self.request_body_json.get('user', {})
        validation.lazy_validate(schema.password_change, user_data)
        try:
            PROVIDERS.identity_api.change_password(user_id=user_id, original_password=user_data['original_password'], new_password=user_data['password'], initiator=self.audit_initiator)
        except AssertionError as e:
            raise ks_exception.Unauthorized(_('Error when changing user password: %s') % e)
        return (None, http.client.NO_CONTENT)