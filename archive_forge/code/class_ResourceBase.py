import datetime
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from werkzeug import exceptions
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception as ks_exceptions
from keystone.i18n import _
from keystone.server import flask as ks_flask
class ResourceBase(ks_flask.ResourceBase):

    @ks_flask.unenforced_api
    def get(self):
        raise exceptions.MethodNotAllowed(valid_methods=['POST'])

    @staticmethod
    def _check_signature(cred_ref, credentials):
        raise NotImplementedError()

    @staticmethod
    def _check_timestamp(credentials):
        timestamp = credentials.get('params', {}).get('Timestamp') or credentials.get('headers', {}).get('X-Amz-Date') or credentials.get('params', {}).get('X-Amz-Date')
        if not timestamp:
            return
        try:
            timestamp = timeutils.parse_isotime(timestamp)
            timestamp = timeutils.normalize_time(timestamp)
        except Exception as e:
            raise ks_exceptions.Unauthorized(_('Credential timestamp is invalid: %s') % e)
        auth_ttl = datetime.timedelta(minutes=CONF.credential.auth_ttl)
        current_time = timeutils.normalize_time(timeutils.utcnow())
        if current_time > timestamp + auth_ttl:
            raise ks_exceptions.Unauthorized(_('Credential is expired'))

    def handle_authenticate(self):
        if not self.request_body_json:
            msg = _('request must include a request body')
            raise ks_exceptions.ValidationError(msg)
        credentials = self.request_body_json.get('credentials') or self.request_body_json.get('credential') or self.request_body_json.get('ec2Credentials')
        if not credentials:
            credentials = {}
        if 'access' not in credentials:
            raise ks_exceptions.Unauthorized(_('EC2 Signature not supplied'))
        credential_id = utils.hash_access_key(credentials['access'])
        cred = PROVIDERS.credential_api.get_credential(credential_id)
        if not cred or cred['type'] != CRED_TYPE_EC2:
            raise ks_exceptions.Unauthorized(_('EC2 access key not found.'))
        try:
            loaded = jsonutils.loads(cred['blob'])
        except TypeError:
            loaded = cred['blob']
        cred_data = dict(user_id=cred.get('user_id'), project_id=cred.get('project_id'), access=loaded.get('access'), secret=loaded.get('secret'), trust_id=loaded.get('trust_id'), app_cred_id=loaded.get('app_cred_id'), access_token_id=loaded.get('access_token_id'))
        self._check_signature(cred_data, credentials)
        project_ref = PROVIDERS.resource_api.get_project(cred_data['project_id'])
        user_ref = PROVIDERS.identity_api.get_user(cred_data['user_id'])
        try:
            PROVIDERS.identity_api.assert_user_enabled(user_id=user_ref['id'], user=user_ref)
            PROVIDERS.resource_api.assert_project_enabled(project_id=project_ref['id'], project=project_ref)
        except AssertionError as e:
            raise ks_exceptions.Unauthorized from e
        self._check_timestamp(credentials)
        trustee_user_id = None
        auth_context = None
        if cred_data['trust_id']:
            trust = PROVIDERS.trust_api.get_trust(cred_data['trust_id'])
            roles = [r['id'] for r in trust['roles']]
            if trust['impersonation'] is True:
                trustee_user_id = trust['trustee_user_id']
        elif cred_data['app_cred_id']:
            ac_client = PROVIDERS.application_credential_api
            app_cred = ac_client.get_application_credential(cred_data['app_cred_id'])
            roles = [r['id'] for r in app_cred['roles']]
        elif cred_data['access_token_id']:
            access_token = PROVIDERS.oauth_api.get_access_token(cred_data['access_token_id'])
            roles = jsonutils.loads(access_token['role_ids'])
            auth_context = {'access_token_id': cred_data['access_token_id']}
        else:
            roles = PROVIDERS.assignment_api.get_roles_for_user_and_project(user_ref['id'], project_ref['id'])
        if not roles:
            raise ks_exceptions.Unauthorized(_('User not valid for project.'))
        for r_id in roles:
            PROVIDERS.role_api.get_role(r_id)
        method_names = ['ec2credential']
        if trustee_user_id:
            user_id = trustee_user_id
        else:
            user_id = user_ref['id']
        token = PROVIDERS.token_provider_api.issue_token(user_id=user_id, method_names=method_names, project_id=project_ref['id'], trust_id=cred_data['trust_id'], app_cred_id=cred_data['app_cred_id'], auth_context=auth_context)
        return token