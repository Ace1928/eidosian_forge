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
class UserOSEC2CredentialsResourceListCreate(_UserOSEC2CredBaseResource):

    def get(self, user_id):
        """List EC2 Credentials for user.

        GET/HEAD /v3/users/{user_id}/credentials/OS-EC2
        """
        ENFORCER.enforce_call(action='identity:ec2_list_credentials')
        PROVIDERS.identity_api.get_user(user_id)
        credential_refs = PROVIDERS.credential_api.list_credentials_for_user(user_id, type=CRED_TYPE_EC2)
        collection_refs = [_convert_v3_to_ec2_credential(cred) for cred in credential_refs]
        return self.wrap_collection(collection_refs)

    def post(self, user_id):
        """Create EC2 Credential for user.

        POST /v3/users/{user_id}/credentials/OS-EC2
        """
        target = {}
        target['credential'] = {'user_id': user_id}
        ENFORCER.enforce_call(action='identity:ec2_create_credential', target_attr=target)
        PROVIDERS.identity_api.get_user(user_id)
        tenant_id = self.request_body_json.get('tenant_id')
        PROVIDERS.resource_api.get_project(tenant_id)
        blob = dict(access=uuid.uuid4().hex, secret=uuid.uuid4().hex, trust_id=self.oslo_context.trust_id)
        credential_id = utils.hash_access_key(blob['access'])
        cred_data = dict(user_id=user_id, project_id=tenant_id, blob=jsonutils.dumps(blob), id=credential_id, type=CRED_TYPE_EC2)
        PROVIDERS.credential_api.create_credential(credential_id, cred_data)
        ref = _convert_v3_to_ec2_credential(cred_data)
        return (self.wrap_member(ref), http.client.CREATED)