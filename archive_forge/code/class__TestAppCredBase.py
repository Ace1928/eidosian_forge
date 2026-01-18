import datetime
import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as base_policy
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _TestAppCredBase(base_classes.TestCaseWithBootstrap):
    """Base class for application credential tests."""

    def _new_app_cred_data(self, user_id=None, project_id=None, name=None, expires=None, system=None):
        if not user_id:
            user_id = self.app_cred_user_id
        if not name:
            name = uuid.uuid4().hex
        if not expires:
            expires = datetime.datetime.utcnow() + datetime.timedelta(days=365)
        if not system:
            system = uuid.uuid4().hex
        if not project_id:
            project_id = self.app_cred_project_id
        app_cred_data = {'id': uuid.uuid4().hex, 'name': name, 'description': uuid.uuid4().hex, 'user_id': user_id, 'project_id': project_id, 'system': system, 'expires_at': expires, 'roles': [{'id': self.bootstrapper.member_role_id}], 'secret': uuid.uuid4().hex, 'unrestricted': False}
        return app_cred_data

    def setUp(self):
        super(_TestAppCredBase, self).setUp()
        new_user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        app_cred_user_ref = PROVIDERS.identity_api.create_user(new_user_ref)
        self.app_cred_user_id = app_cred_user_ref['id']
        self.app_cred_user_password = new_user_ref['password']
        app_cred_project_ref = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        self.app_cred_project_id = app_cred_project_ref['id']
        PROVIDERS.assignment_api.create_grant(self.bootstrapper.member_role_id, user_id=self.app_cred_user_id, project_id=self.app_cred_project_id)

    def _create_application_credential(self):
        app_cred = self._new_app_cred_data()
        return PROVIDERS.application_credential_api.create_application_credential(app_cred)

    def _override_policy(self):
        with open(self.policy_file_name, 'w') as f:
            overridden_policies = {'identity:get_application_credential': base_policy.RULE_SYSTEM_READER_OR_OWNER, 'identity:list_application_credentials': base_policy.RULE_SYSTEM_READER_OR_OWNER, 'identity:create_application_credential': base_policy.RULE_OWNER, 'identity:delete_application_credential': base_policy.RULE_SYSTEM_ADMIN_OR_OWNER}
            f.write(jsonutils.dumps(overridden_policies))