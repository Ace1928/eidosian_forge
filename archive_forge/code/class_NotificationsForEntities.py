import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
class NotificationsForEntities(BaseNotificationTest):

    def test_create_group(self):
        group_ref = unit.new_group_ref(domain_id=self.domain_id)
        group_ref = PROVIDERS.identity_api.create_group(group_ref)
        self._assert_last_note(group_ref['id'], CREATED_OPERATION, 'group')
        self._assert_last_audit(group_ref['id'], CREATED_OPERATION, 'group', cadftaxonomy.SECURITY_GROUP)

    def test_create_project(self):
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        self._assert_last_note(project_ref['id'], CREATED_OPERATION, 'project')
        self._assert_last_audit(project_ref['id'], CREATED_OPERATION, 'project', cadftaxonomy.SECURITY_PROJECT)

    def test_create_role(self):
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        self._assert_last_note(role_ref['id'], CREATED_OPERATION, 'role')
        self._assert_last_audit(role_ref['id'], CREATED_OPERATION, 'role', cadftaxonomy.SECURITY_ROLE)

    def test_create_user(self):
        user_ref = unit.new_user_ref(domain_id=self.domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        self._assert_last_note(user_ref['id'], CREATED_OPERATION, 'user')
        self._assert_last_audit(user_ref['id'], CREATED_OPERATION, 'user', cadftaxonomy.SECURITY_ACCOUNT_USER)

    def test_create_trust(self):
        trustor = unit.new_user_ref(domain_id=self.domain_id)
        trustor = PROVIDERS.identity_api.create_user(trustor)
        trustee = unit.new_user_ref(domain_id=self.domain_id)
        trustee = PROVIDERS.identity_api.create_user(trustee)
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        trust_ref = unit.new_trust_ref(trustor['id'], trustee['id'])
        PROVIDERS.trust_api.create_trust(trust_ref['id'], trust_ref, [role_ref])
        self._assert_last_note(trust_ref['id'], CREATED_OPERATION, 'OS-TRUST:trust')
        self._assert_last_audit(trust_ref['id'], CREATED_OPERATION, 'OS-TRUST:trust', cadftaxonomy.SECURITY_TRUST)

    def test_delete_group(self):
        group_ref = unit.new_group_ref(domain_id=self.domain_id)
        group_ref = PROVIDERS.identity_api.create_group(group_ref)
        PROVIDERS.identity_api.delete_group(group_ref['id'])
        self._assert_last_note(group_ref['id'], DELETED_OPERATION, 'group')
        self._assert_last_audit(group_ref['id'], DELETED_OPERATION, 'group', cadftaxonomy.SECURITY_GROUP)

    def test_delete_project(self):
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        PROVIDERS.resource_api.delete_project(project_ref['id'])
        self._assert_last_note(project_ref['id'], DELETED_OPERATION, 'project')
        self._assert_last_audit(project_ref['id'], DELETED_OPERATION, 'project', cadftaxonomy.SECURITY_PROJECT)

    def test_delete_role(self):
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        PROVIDERS.role_api.delete_role(role_ref['id'])
        self._assert_last_note(role_ref['id'], DELETED_OPERATION, 'role')
        self._assert_last_audit(role_ref['id'], DELETED_OPERATION, 'role', cadftaxonomy.SECURITY_ROLE)

    def test_delete_user(self):
        user_ref = unit.new_user_ref(domain_id=self.domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        PROVIDERS.identity_api.delete_user(user_ref['id'])
        self._assert_last_note(user_ref['id'], DELETED_OPERATION, 'user')
        self._assert_last_audit(user_ref['id'], DELETED_OPERATION, 'user', cadftaxonomy.SECURITY_ACCOUNT_USER)

    def test_create_domain(self):
        domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        self._assert_last_note(domain_ref['id'], CREATED_OPERATION, 'domain')
        self._assert_last_audit(domain_ref['id'], CREATED_OPERATION, 'domain', cadftaxonomy.SECURITY_DOMAIN)

    def test_update_domain(self):
        domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        domain_ref['description'] = uuid.uuid4().hex
        PROVIDERS.resource_api.update_domain(domain_ref['id'], domain_ref)
        self._assert_last_note(domain_ref['id'], UPDATED_OPERATION, 'domain')
        self._assert_last_audit(domain_ref['id'], UPDATED_OPERATION, 'domain', cadftaxonomy.SECURITY_DOMAIN)

    def test_delete_domain(self):
        domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        domain_ref['enabled'] = False
        PROVIDERS.resource_api.update_domain(domain_ref['id'], domain_ref)
        PROVIDERS.resource_api.delete_domain(domain_ref['id'])
        self._assert_last_note(domain_ref['id'], DELETED_OPERATION, 'domain')
        self._assert_last_audit(domain_ref['id'], DELETED_OPERATION, 'domain', cadftaxonomy.SECURITY_DOMAIN)

    def test_delete_trust(self):
        trustor = unit.new_user_ref(domain_id=self.domain_id)
        trustor = PROVIDERS.identity_api.create_user(trustor)
        trustee = unit.new_user_ref(domain_id=self.domain_id)
        trustee = PROVIDERS.identity_api.create_user(trustee)
        role_ref = unit.new_role_ref()
        trust_ref = unit.new_trust_ref(trustor['id'], trustee['id'])
        PROVIDERS.trust_api.create_trust(trust_ref['id'], trust_ref, [role_ref])
        PROVIDERS.trust_api.delete_trust(trust_ref['id'])
        self._assert_last_note(trust_ref['id'], DELETED_OPERATION, 'OS-TRUST:trust')
        self._assert_last_audit(trust_ref['id'], DELETED_OPERATION, 'OS-TRUST:trust', cadftaxonomy.SECURITY_TRUST)

    def test_create_endpoint(self):
        endpoint_ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
        PROVIDERS.catalog_api.create_endpoint(endpoint_ref['id'], endpoint_ref)
        self._assert_notify_sent(endpoint_ref['id'], CREATED_OPERATION, 'endpoint')
        self._assert_last_audit(endpoint_ref['id'], CREATED_OPERATION, 'endpoint', cadftaxonomy.SECURITY_ENDPOINT)

    def test_update_endpoint(self):
        endpoint_ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
        PROVIDERS.catalog_api.create_endpoint(endpoint_ref['id'], endpoint_ref)
        PROVIDERS.catalog_api.update_endpoint(endpoint_ref['id'], endpoint_ref)
        self._assert_notify_sent(endpoint_ref['id'], UPDATED_OPERATION, 'endpoint')
        self._assert_last_audit(endpoint_ref['id'], UPDATED_OPERATION, 'endpoint', cadftaxonomy.SECURITY_ENDPOINT)

    def test_delete_endpoint(self):
        endpoint_ref = unit.new_endpoint_ref(service_id=self.service_id, interface='public', region_id=self.region_id)
        PROVIDERS.catalog_api.create_endpoint(endpoint_ref['id'], endpoint_ref)
        PROVIDERS.catalog_api.delete_endpoint(endpoint_ref['id'])
        self._assert_notify_sent(endpoint_ref['id'], DELETED_OPERATION, 'endpoint')
        self._assert_last_audit(endpoint_ref['id'], DELETED_OPERATION, 'endpoint', cadftaxonomy.SECURITY_ENDPOINT)

    def test_create_service(self):
        service_ref = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service_ref['id'], service_ref)
        self._assert_notify_sent(service_ref['id'], CREATED_OPERATION, 'service')
        self._assert_last_audit(service_ref['id'], CREATED_OPERATION, 'service', cadftaxonomy.SECURITY_SERVICE)

    def test_update_service(self):
        service_ref = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service_ref['id'], service_ref)
        PROVIDERS.catalog_api.update_service(service_ref['id'], service_ref)
        self._assert_notify_sent(service_ref['id'], UPDATED_OPERATION, 'service')
        self._assert_last_audit(service_ref['id'], UPDATED_OPERATION, 'service', cadftaxonomy.SECURITY_SERVICE)

    def test_delete_service(self):
        service_ref = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(service_ref['id'], service_ref)
        PROVIDERS.catalog_api.delete_service(service_ref['id'])
        self._assert_notify_sent(service_ref['id'], DELETED_OPERATION, 'service')
        self._assert_last_audit(service_ref['id'], DELETED_OPERATION, 'service', cadftaxonomy.SECURITY_SERVICE)

    def test_create_region(self):
        region_ref = unit.new_region_ref()
        PROVIDERS.catalog_api.create_region(region_ref)
        self._assert_notify_sent(region_ref['id'], CREATED_OPERATION, 'region')
        self._assert_last_audit(region_ref['id'], CREATED_OPERATION, 'region', cadftaxonomy.SECURITY_REGION)

    def test_update_region(self):
        region_ref = unit.new_region_ref()
        PROVIDERS.catalog_api.create_region(region_ref)
        PROVIDERS.catalog_api.update_region(region_ref['id'], region_ref)
        self._assert_notify_sent(region_ref['id'], UPDATED_OPERATION, 'region')
        self._assert_last_audit(region_ref['id'], UPDATED_OPERATION, 'region', cadftaxonomy.SECURITY_REGION)

    def test_delete_region(self):
        region_ref = unit.new_region_ref()
        PROVIDERS.catalog_api.create_region(region_ref)
        PROVIDERS.catalog_api.delete_region(region_ref['id'])
        self._assert_notify_sent(region_ref['id'], DELETED_OPERATION, 'region')
        self._assert_last_audit(region_ref['id'], DELETED_OPERATION, 'region', cadftaxonomy.SECURITY_REGION)

    def test_create_policy(self):
        policy_ref = unit.new_policy_ref()
        PROVIDERS.policy_api.create_policy(policy_ref['id'], policy_ref)
        self._assert_notify_sent(policy_ref['id'], CREATED_OPERATION, 'policy')
        self._assert_last_audit(policy_ref['id'], CREATED_OPERATION, 'policy', cadftaxonomy.SECURITY_POLICY)

    def test_update_policy(self):
        policy_ref = unit.new_policy_ref()
        PROVIDERS.policy_api.create_policy(policy_ref['id'], policy_ref)
        PROVIDERS.policy_api.update_policy(policy_ref['id'], policy_ref)
        self._assert_notify_sent(policy_ref['id'], UPDATED_OPERATION, 'policy')
        self._assert_last_audit(policy_ref['id'], UPDATED_OPERATION, 'policy', cadftaxonomy.SECURITY_POLICY)

    def test_delete_policy(self):
        policy_ref = unit.new_policy_ref()
        PROVIDERS.policy_api.create_policy(policy_ref['id'], policy_ref)
        PROVIDERS.policy_api.delete_policy(policy_ref['id'])
        self._assert_notify_sent(policy_ref['id'], DELETED_OPERATION, 'policy')
        self._assert_last_audit(policy_ref['id'], DELETED_OPERATION, 'policy', cadftaxonomy.SECURITY_POLICY)

    def test_disable_domain(self):
        domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        domain_ref['enabled'] = False
        PROVIDERS.resource_api.update_domain(domain_ref['id'], domain_ref)
        self._assert_notify_sent(domain_ref['id'], 'disabled', 'domain', public=False)

    def test_disable_of_disabled_domain_does_not_notify(self):
        domain_ref = unit.new_domain_ref(enabled=False)
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        PROVIDERS.resource_api.update_domain(domain_ref['id'], domain_ref)
        self._assert_notify_not_sent(domain_ref['id'], 'disabled', 'domain', public=False)

    def test_update_group(self):
        group_ref = unit.new_group_ref(domain_id=self.domain_id)
        group_ref = PROVIDERS.identity_api.create_group(group_ref)
        PROVIDERS.identity_api.update_group(group_ref['id'], group_ref)
        self._assert_last_note(group_ref['id'], UPDATED_OPERATION, 'group')
        self._assert_last_audit(group_ref['id'], UPDATED_OPERATION, 'group', cadftaxonomy.SECURITY_GROUP)

    def test_update_project(self):
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        PROVIDERS.resource_api.update_project(project_ref['id'], project_ref)
        self._assert_notify_sent(project_ref['id'], UPDATED_OPERATION, 'project', public=True)
        self._assert_last_audit(project_ref['id'], UPDATED_OPERATION, 'project', cadftaxonomy.SECURITY_PROJECT)

    def test_disable_project(self):
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        project_ref['enabled'] = False
        PROVIDERS.resource_api.update_project(project_ref['id'], project_ref)
        self._assert_notify_sent(project_ref['id'], 'disabled', 'project', public=False)

    def test_disable_of_disabled_project_does_not_notify(self):
        project_ref = unit.new_project_ref(domain_id=self.domain_id, enabled=False)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        PROVIDERS.resource_api.update_project(project_ref['id'], project_ref)
        self._assert_notify_not_sent(project_ref['id'], 'disabled', 'project', public=False)

    def test_update_project_does_not_send_disable(self):
        project_ref = unit.new_project_ref(domain_id=self.domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        project_ref['enabled'] = True
        PROVIDERS.resource_api.update_project(project_ref['id'], project_ref)
        self._assert_last_note(project_ref['id'], UPDATED_OPERATION, 'project')
        self._assert_notify_not_sent(project_ref['id'], 'disabled', 'project')

    def test_update_role(self):
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        PROVIDERS.role_api.update_role(role_ref['id'], role_ref)
        self._assert_last_note(role_ref['id'], UPDATED_OPERATION, 'role')
        self._assert_last_audit(role_ref['id'], UPDATED_OPERATION, 'role', cadftaxonomy.SECURITY_ROLE)

    def test_update_user(self):
        user_ref = unit.new_user_ref(domain_id=self.domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        PROVIDERS.identity_api.update_user(user_ref['id'], user_ref)
        self._assert_last_note(user_ref['id'], UPDATED_OPERATION, 'user')
        self._assert_last_audit(user_ref['id'], UPDATED_OPERATION, 'user', cadftaxonomy.SECURITY_ACCOUNT_USER)

    def test_config_option_no_events(self):
        self.config_fixture.config(notification_format='basic')
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        self._assert_last_note(role_ref['id'], CREATED_OPERATION, 'role')
        self.assertEqual(0, len(self._audits))

    def test_add_user_to_group(self):
        user_ref = unit.new_user_ref(domain_id=self.domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        group_ref = unit.new_group_ref(domain_id=self.domain_id)
        group_ref = PROVIDERS.identity_api.create_group(group_ref)
        PROVIDERS.identity_api.add_user_to_group(user_ref['id'], group_ref['id'])
        self._assert_last_note(group_ref['id'], UPDATED_OPERATION, 'group', actor_id=user_ref['id'], actor_type='user', actor_operation='added')

    def test_remove_user_from_group(self):
        user_ref = unit.new_user_ref(domain_id=self.domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        group_ref = unit.new_group_ref(domain_id=self.domain_id)
        group_ref = PROVIDERS.identity_api.create_group(group_ref)
        PROVIDERS.identity_api.add_user_to_group(user_ref['id'], group_ref['id'])
        PROVIDERS.identity_api.remove_user_from_group(user_ref['id'], group_ref['id'])
        self._assert_last_note(group_ref['id'], UPDATED_OPERATION, 'group', actor_id=user_ref['id'], actor_type='user', actor_operation='removed')

    def test_initiator_request_id(self):
        ref = unit.new_domain_ref()
        self.post('/domains', body={'domain': ref})
        note = self._notifications[-1]
        initiator = note['initiator']
        self.assertIsNotNone(initiator.request_id)

    def test_initiator_global_request_id(self):
        global_request_id = 'req-%s' % uuid.uuid4()
        ref = unit.new_domain_ref()
        self.post('/domains', body={'domain': ref}, headers={'X-OpenStack-Request-Id': global_request_id})
        note = self._notifications[-1]
        initiator = note['initiator']
        self.assertEqual(initiator.global_request_id, global_request_id)

    def test_initiator_global_request_id_not_set(self):
        ref = unit.new_domain_ref()
        self.post('/domains', body={'domain': ref})
        note = self._notifications[-1]
        initiator = note['initiator']
        self.assertFalse(hasattr(initiator, 'global_request_id'))