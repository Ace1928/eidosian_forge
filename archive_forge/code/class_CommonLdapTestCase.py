import os
import tempfile
from unittest import mock
import uuid
import fixtures
import ldap.dn
from oslo_config import fixture as config_fixture
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception as ks_exception
from keystone.identity.backends.ldap import common as common_ldap
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import fakeldap
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
class CommonLdapTestCase(unit.BaseTestCase):
    """These test cases call functions in keystone.common.ldap."""

    def test_binary_attribute_values(self):
        result = [('cn=junk,dc=example,dc=com', {'cn': ['junk'], 'sn': [uuid.uuid4().hex], 'mail': [uuid.uuid4().hex], 'binary_attr': [b'\x00\xff\x00\xff']})]
        py_result = common_ldap.convert_ldap_result(result)
        self.assertNotIn('binary_attr', py_result[0][1])

    def test_utf8_conversion(self):
        value_unicode = u'fäké1'
        value_utf8 = value_unicode.encode('utf-8')
        result_utf8 = common_ldap.utf8_encode(value_unicode)
        self.assertEqual(value_utf8, result_utf8)
        result_utf8 = common_ldap.utf8_encode(value_utf8)
        self.assertEqual(value_utf8, result_utf8)
        result_unicode = common_ldap.utf8_decode(value_utf8)
        self.assertEqual(value_unicode, result_unicode)
        result_unicode = common_ldap.utf8_decode(value_unicode)
        self.assertEqual(value_unicode, result_unicode)
        self.assertRaises(TypeError, common_ldap.utf8_encode, 100)
        result_unicode = common_ldap.utf8_decode(100)
        self.assertEqual(u'100', result_unicode)

    def test_user_id_begins_with_0(self):
        user_id = '0123456'
        result = [('cn=dummy,dc=example,dc=com', {'user_id': [user_id], 'enabled': ['TRUE']})]
        py_result = common_ldap.convert_ldap_result(result)
        self.assertIs(True, py_result[0][1]['enabled'][0])
        self.assertEqual(user_id, py_result[0][1]['user_id'][0])

    def test_user_id_begins_with_0_and_enabled_bit_mask(self):
        user_id = '0123456'
        bitmask = '225'
        expected_bitmask = 225
        result = [('cn=dummy,dc=example,dc=com', {'user_id': [user_id], 'enabled': [bitmask]})]
        py_result = common_ldap.convert_ldap_result(result)
        self.assertEqual(expected_bitmask, py_result[0][1]['enabled'][0])
        self.assertEqual(user_id, py_result[0][1]['user_id'][0])

    def test_user_id_and_bitmask_begins_with_0(self):
        user_id = '0123456'
        bitmask = '0225'
        expected_bitmask = 225
        result = [('cn=dummy,dc=example,dc=com', {'user_id': [user_id], 'enabled': [bitmask]})]
        py_result = common_ldap.convert_ldap_result(result)
        self.assertEqual(expected_bitmask, py_result[0][1]['enabled'][0])
        self.assertEqual(user_id, py_result[0][1]['user_id'][0])

    def test_user_id_and_user_name_with_boolean_string(self):
        boolean_strings = ['TRUE', 'FALSE', 'true', 'false', 'True', 'False', 'TrUeFaLse']
        for user_name in boolean_strings:
            user_id = uuid.uuid4().hex
            result = [('cn=dummy,dc=example,dc=com', {'user_id': [user_id], 'user_name': [user_name]})]
            py_result = common_ldap.convert_ldap_result(result)
            self.assertEqual(user_name, py_result[0][1]['user_name'][0])

    def test_user_id_attribute_is_uuid_in_byte_form(self):
        results = [('cn=alice,dc=example,dc=com', {'cn': [b'cn=alice'], 'objectGUID': [b'\xdd\xd8Rt\xee]bA\x8e(\xe39\x0b\xe1\xf8\xe8'], 'email': [uuid.uuid4().hex], 'sn': [uuid.uuid4().hex]})]
        py_result = common_ldap.convert_ldap_result(results)
        exp_object_guid = '7452d8dd-5dee-4162-8e28-e3390be1f8e8'
        self.assertEqual(exp_object_guid, py_result[0][1]['objectGUID'][0])