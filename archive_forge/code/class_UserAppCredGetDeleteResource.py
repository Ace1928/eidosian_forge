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
class UserAppCredGetDeleteResource(ks_flask.ResourceBase):
    collection_key = 'application_credentials'
    member_key = 'application_credential'

    def get(self, user_id, application_credential_id):
        """Get application credential resource.

        GET/HEAD /v3/users/{user_id}/application_credentials/
                 {application_credential_id}
        """
        target = _update_request_user_id_attribute()
        ENFORCER.enforce_call(action='identity:get_application_credential', target_attr=target)
        ref = PROVIDERS.application_credential_api.get_application_credential(application_credential_id)
        return self.wrap_member(ref)

    def delete(self, user_id, application_credential_id):
        """Delete application credential resource.

        DELETE /v3/users/{user_id}/application_credentials/
               {application_credential_id}
        """
        target = _update_request_user_id_attribute()
        ENFORCER.enforce_call(action='identity:delete_application_credential', target_attr=target)
        token = self.auth_context['token']
        _check_unrestricted_application_credential(token)
        PROVIDERS.application_credential_api.delete_application_credential(application_credential_id, initiator=self.audit_initiator)
        return (None, http.client.NO_CONTENT)