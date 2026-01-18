import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
class UserFederatedAttributesTests(test_v3.RestfulTestCase):

    def _create_federated_attributes(self):
        idp = {'id': uuid.uuid4().hex, 'enabled': True, 'description': uuid.uuid4().hex}
        PROVIDERS.federation_api.create_idp(idp['id'], idp)
        mapping = mapping_fixtures.MAPPING_EPHEMERAL_USER
        mapping['id'] = uuid.uuid4().hex
        PROVIDERS.federation_api.create_mapping(mapping['id'], mapping)
        protocol = {'id': uuid.uuid4().hex, 'mapping_id': mapping['id']}
        PROVIDERS.federation_api.create_protocol(idp['id'], protocol['id'], protocol)
        return (idp, protocol)

    def _create_user_with_federated_user(self, user, fed_dict):
        with sql.session_for_write() as session:
            federated_ref = model.FederatedUser.from_dict(fed_dict)
            user_ref = model.User.from_dict(user)
            user_ref.created_at = datetime.datetime.utcnow()
            user_ref.federated_users.append(federated_ref)
            session.add(user_ref)
            return identity_base.filter_user(user_ref.to_dict())

    def setUp(self):
        super(UserFederatedAttributesTests, self).setUp()
        self.useFixture(database.Database())
        self.load_backends()
        idp, protocol = self._create_federated_attributes()
        self.fed_dict = unit.new_federated_user_ref()
        self.fed_dict['idp_id'] = idp['id']
        self.fed_dict['protocol_id'] = protocol['id']
        self.fed_dict['unique_id'] = 'jdoe'
        self.domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domain['id'], self.domain)
        self.fed_user = unit.new_user_ref(domain_id=self.domain['id'])
        self.fed_user = self._create_user_with_federated_user(self.fed_user, self.fed_dict)
        idp, protocol = self._create_federated_attributes()
        self.fed_dict2 = unit.new_federated_user_ref()
        self.fed_dict2['idp_id'] = idp['id']
        self.fed_dict2['protocol_id'] = protocol['id']
        self.fed_dict2['unique_id'] = 'ravelar'
        self.fed_user2 = unit.new_user_ref(domain_id=self.domain['id'])
        self.fed_user2 = self._create_user_with_federated_user(self.fed_user2, self.fed_dict2)
        self.fed_dict3 = unit.new_federated_user_ref()
        self.fed_dict3['idp_id'] = idp['id']
        self.fed_dict3['protocol_id'] = protocol['id']
        self.fed_dict3['unique_id'] = 'jsmith'
        self.fed_user3 = unit.new_user_ref(domain_id=self.domain['id'])
        self.fed_user3 = self._create_user_with_federated_user(self.fed_user3, self.fed_dict3)

    def _test_list_users_with_federated_parameter(self, parameter):
        resource_url = '/users?%s=%s' % (parameter[0], self.fed_dict[parameter[0]])
        for attr in parameter[1:]:
            resource_url += '&%s=%s' % (attr, self.fed_dict[attr])
        r = self.get(resource_url)
        self.assertEqual(1, len(r.result['users']))
        self.assertValidUserListResponse(r, ref=self.fed_user, resource_url=resource_url)
        if not any(('unique_id' in x for x in parameter)):
            resource_url = '/users?%s=%s' % (parameter[0], self.fed_dict2[parameter[0]])
            for attr in parameter[1:]:
                resource_url += '&%s=%s' % (attr, self.fed_dict2[attr])
            r = self.get(resource_url)
            self.assertEqual(2, len(r.result['users']))
            self.assertValidUserListResponse(r, ref=self.fed_user2, resource_url=resource_url)

    def test_list_users_with_idp_id(self):
        attribute = ['idp_id']
        self._test_list_users_with_federated_parameter(attribute)

    def test_list_users_with_protocol_id(self):
        attribute = ['protocol_id']
        self._test_list_users_with_federated_parameter(attribute)

    def test_list_users_with_unique_id(self):
        attribute = ['unique_id']
        self._test_list_users_with_federated_parameter(attribute)

    def test_list_users_with_idp_id_and_unique_id(self):
        attribute = ['idp_id', 'unique_id']
        self._test_list_users_with_federated_parameter(attribute)

    def test_list_users_with_idp_id_and_protocol_id(self):
        attribute = ['idp_id', 'protocol_id']
        self._test_list_users_with_federated_parameter(attribute)

    def test_list_users_with_protocol_id_and_unique_id(self):
        attribute = ['protocol_id', 'unique_id']
        self._test_list_users_with_federated_parameter(attribute)

    def test_list_users_with_all_federated_attributes(self):
        attribute = ['idp_id', 'protocol_id', 'unique_id']
        self._test_list_users_with_federated_parameter(attribute)

    def test_get_user_includes_required_federated_attributes(self):
        user = self.identity_api.get_user(self.fed_user['id'])
        self.assertIn('federated', user)
        self.assertIn('idp_id', user['federated'][0])
        self.assertIn('protocols', user['federated'][0])
        self.assertIn('protocol_id', user['federated'][0]['protocols'][0])
        self.assertIn('unique_id', user['federated'][0]['protocols'][0])
        r = self.get('/users/%(user_id)s' % {'user_id': user['id']})
        self.assertValidUserResponse(r, user)

    def test_create_user_with_federated_attributes(self):
        """Call ``POST /users``."""
        idp, protocol = self._create_federated_attributes()
        ref = unit.new_user_ref(domain_id=self.domain_id)
        ref['federated'] = [{'idp_id': idp['id'], 'protocols': [{'protocol_id': protocol['id'], 'unique_id': uuid.uuid4().hex}]}]
        r = self.post('/users', body={'user': ref})
        user = r.result['user']
        self.assertEqual(user['name'], ref['name'])
        self.assertEqual(user['federated'], ref['federated'])
        self.assertValidUserResponse(r, ref)

    def test_create_user_fails_when_given_invalid_idp_and_protocols(self):
        """Call ``POST /users`` with invalid idp and protocol to fail."""
        idp, protocol = self._create_federated_attributes()
        ref = unit.new_user_ref(domain_id=self.domain_id)
        ref['federated'] = [{'idp_id': 'fakeidp', 'protocols': [{'protocol_id': 'fakeprotocol_id', 'unique_id': uuid.uuid4().hex}]}]
        self.post('/users', body={'user': ref}, token=self.get_admin_token(), expected_status=http.client.BAD_REQUEST)
        ref['federated'][0]['idp_id'] = idp['id']
        self.post('/users', body={'user': ref}, token=self.get_admin_token(), expected_status=http.client.BAD_REQUEST)

    def test_update_user_with_federated_attributes(self):
        """Call ``PATCH /users/{user_id}``."""
        user = self.fed_user.copy()
        del user['id']
        user['name'] = 'James Doe'
        idp, protocol = self._create_federated_attributes()
        user['federated'] = [{'idp_id': idp['id'], 'protocols': [{'protocol_id': protocol['id'], 'unique_id': 'jdoe'}]}]
        r = self.patch('/users/%(user_id)s' % {'user_id': self.fed_user['id']}, body={'user': user})
        resp_user = r.result['user']
        self.assertEqual(user['name'], resp_user['name'])
        self.assertEqual(user['federated'], resp_user['federated'])
        self.assertValidUserResponse(r, user)

    def test_update_user_fails_when_given_invalid_idp_and_protocols(self):
        """Call ``PATCH /users/{user_id}``."""
        user = self.fed_user.copy()
        del user['id']
        idp, protocol = self._create_federated_attributes()
        user['federated'] = [{'idp_id': 'fakeidp', 'protocols': [{'protocol_id': 'fakeprotocol_id', 'unique_id': uuid.uuid4().hex}]}]
        self.patch('/users/%(user_id)s' % {'user_id': self.fed_user['id']}, body={'user': user}, expected_status=http.client.BAD_REQUEST)
        user['federated'][0]['idp_id'] = idp['id']
        self.patch('/users/%(user_id)s' % {'user_id': self.fed_user['id']}, body={'user': user}, expected_status=http.client.BAD_REQUEST)