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
class UserOSEC2CredentialsResourceGetDelete(_UserOSEC2CredBaseResource):

    @staticmethod
    def _get_cred_data(credential_id):
        cred = PROVIDERS.credential_api.get_credential(credential_id)
        if not cred or cred['type'] != CRED_TYPE_EC2:
            raise ks_exception.Unauthorized(message=_('EC2 access key not found.'))
        return _convert_v3_to_ec2_credential(cred)

    def get(self, user_id, credential_id):
        """Get a specific EC2 credential.

        GET/HEAD /users/{user_id}/credentials/OS-EC2/{credential_id}
        """
        func = _build_enforcer_target_data_owner_and_user_id_match
        ENFORCER.enforce_call(action='identity:ec2_get_credential', build_target=func)
        PROVIDERS.identity_api.get_user(user_id)
        ec2_cred_id = utils.hash_access_key(credential_id)
        cred_data = self._get_cred_data(ec2_cred_id)
        return self.wrap_member(cred_data)

    def delete(self, user_id, credential_id):
        """Delete a specific EC2 credential.

        DELETE /users/{user_id}/credentials/OS-EC2/{credential_id}
        """
        func = _build_enforcer_target_data_owner_and_user_id_match
        ENFORCER.enforce_call(action='identity:ec2_delete_credential', build_target=func)
        PROVIDERS.identity_api.get_user(user_id)
        ec2_cred_id = utils.hash_access_key(credential_id)
        self._get_cred_data(ec2_cred_id)
        PROVIDERS.credential_api.delete_credential(ec2_cred_id)
        return (None, http.client.NO_CONTENT)