import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
class DomainConfigTests(object):

    def setUp(self):
        self.domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domain['id'], self.domain)
        self.addCleanup(self.clean_up_domain)

    def clean_up_domain(self):
        self.domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(self.domain['id'], self.domain)
        PROVIDERS.resource_api.delete_domain(self.domain['id'])
        del self.domain

    def test_create_domain_config_including_sensitive_option(self):
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
        config_whitelisted = copy.deepcopy(config)
        config_whitelisted['ldap'].pop('password')
        self.assertEqual(config_whitelisted, res)
        res = PROVIDERS.domain_config_api.driver.get_config_option(self.domain['id'], 'ldap', 'password', sensitive=True)
        self.assertEqual(config['ldap']['password'], res['value'])
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertEqual(config, res)

    def test_get_partial_domain_config(self):
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'], group='identity')
        config_partial = copy.deepcopy(config)
        config_partial.pop('ldap')
        self.assertEqual(config_partial, res)
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'], group='ldap', option='user_tree_dn')
        self.assertEqual({'user_tree_dn': config['ldap']['user_tree_dn']}, res)
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, self.domain['id'], group='ldap', option='password')

    def test_delete_partial_domain_config(self):
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        PROVIDERS.domain_config_api.delete_config(self.domain['id'], group='identity')
        config_partial = copy.deepcopy(config)
        config_partial.pop('identity')
        config_partial['ldap'].pop('password')
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
        self.assertEqual(config_partial, res)
        PROVIDERS.domain_config_api.delete_config(self.domain['id'], group='ldap', option='url')
        config_partial = copy.deepcopy(config_partial)
        config_partial['ldap'].pop('url')
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
        self.assertEqual(config_partial, res)

    def test_get_options_not_in_domain_config(self):
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, self.domain['id'])
        config = {'ldap': {'url': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, self.domain['id'], group='identity')
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, self.domain['id'], group='ldap', option='user_tree_dn')

    def test_get_sensitive_config(self):
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertEqual({}, res)
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertEqual(config, res)

    def test_update_partial_domain_config(self):
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        new_config = {'ldap': {'url': uuid.uuid4().hex, 'user_filter': uuid.uuid4().hex}}
        res = PROVIDERS.domain_config_api.update_config(self.domain['id'], new_config, group='ldap')
        expected_config = copy.deepcopy(config)
        expected_config['ldap']['url'] = new_config['ldap']['url']
        expected_config['ldap']['user_filter'] = new_config['ldap']['user_filter']
        expected_full_config = copy.deepcopy(expected_config)
        expected_config['ldap'].pop('password')
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
        self.assertEqual(expected_config, res)
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertEqual(expected_full_config, res)
        PROVIDERS.domain_config_api.delete_config(self.domain['id'])
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        new_config = {'url': uuid.uuid4().hex}
        res = PROVIDERS.domain_config_api.update_config(self.domain['id'], new_config, group='ldap', option='url')
        expected_whitelisted_config = copy.deepcopy(config)
        expected_whitelisted_config['ldap']['url'] = new_config['url']
        expected_full_config = copy.deepcopy(expected_whitelisted_config)
        expected_whitelisted_config['ldap'].pop('password')
        self.assertEqual(expected_whitelisted_config, res)
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
        self.assertEqual(expected_whitelisted_config, res)
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertEqual(expected_full_config, res)
        PROVIDERS.domain_config_api.delete_config(self.domain['id'])
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        new_config = {'password': uuid.uuid4().hex}
        res = PROVIDERS.domain_config_api.update_config(self.domain['id'], new_config, group='ldap', option='password')
        expected_whitelisted_config = copy.deepcopy(config)
        expected_full_config = copy.deepcopy(config)
        expected_whitelisted_config['ldap'].pop('password')
        self.assertEqual(expected_whitelisted_config, res)
        res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
        self.assertEqual(expected_whitelisted_config, res)
        expected_full_config['ldap']['password'] = new_config['password']
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertEqual(expected_full_config, res)

    def test_update_invalid_partial_domain_config(self):
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group='ldap')
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config['ldap'], group='ldap', option='url')
        config = {'ldap': {'user_tree_dn': uuid.uuid4().hex}}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group='identity')
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config['ldap'], group='ldap', option='url')
        config = {'ldap': {'user_tree_dn': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        config_wrong_group = {'identity': {'driver': uuid.uuid4().hex}}
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.update_config, self.domain['id'], config_wrong_group, group='identity')
        config_wrong_option = {'url': uuid.uuid4().hex}
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.update_config, self.domain['id'], config_wrong_option, group='ldap', option='url')
        bad_group = uuid.uuid4().hex
        config = {bad_group: {'user': uuid.uuid4().hex}}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group=bad_group, option='user')
        bad_option = uuid.uuid4().hex
        config = {'ldap': {bad_option: uuid.uuid4().hex}}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.update_config, self.domain['id'], config, group='ldap', option=bad_option)

    def test_create_invalid_domain_config(self):
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], {})
        config = {uuid.uuid4().hex: uuid.uuid4().hex}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)
        config = {uuid.uuid4().hex: {uuid.uuid4().hex: uuid.uuid4().hex}}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)
        config = {'ldap': {uuid.uuid4().hex: uuid.uuid4().hex}}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)
        config = {'identity': {'user_tree_dn': uuid.uuid4().hex}}
        self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)

    def test_delete_invalid_partial_domain_config(self):
        config = {'ldap': {'url': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.delete_config, self.domain['id'], group='identity')
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.delete_config, self.domain['id'], group='ldap', option='user_tree_dn')

    def test_sensitive_substitution_in_domain_config(self):
        config = {'ldap': {'url': 'my_url/%(password)s', 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        expected_url = config['ldap']['url'] % {'password': config['ldap']['password']}
        self.assertEqual(expected_url, res['ldap']['url'])

    def test_invalid_sensitive_substitution_in_domain_config(self):
        """Check that invalid substitutions raise warnings."""
        mock_log = mock.Mock()
        invalid_option_config = {'ldap': {'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        for invalid_option in ['my_url/%(passssword)s', 'my_url/%(password', 'my_url/%(password)', 'my_url/%(password)d']:
            invalid_option_config['ldap']['url'] = invalid_option
            PROVIDERS.domain_config_api.create_config(self.domain['id'], invalid_option_config)
            with mock.patch('keystone.resource.core.LOG', mock_log):
                res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
            mock_log.warning.assert_any_call(mock.ANY, mock.ANY)
            self.assertEqual(invalid_option_config['ldap']['url'], res['ldap']['url'])

    def test_escaped_sequence_in_domain_config(self):
        """Check that escaped '%(' doesn't get interpreted."""
        mock_log = mock.Mock()
        escaped_option_config = {'ldap': {'url': 'my_url/%%(password)s', 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], escaped_option_config)
        with mock.patch('keystone.resource.core.LOG', mock_log):
            res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertFalse(mock_log.warn.called)
        self.assertEqual('my_url/%(password)s', res['ldap']['url'])

    @unit.skip_if_cache_disabled('domain_config')
    def test_cache_layer_get_sensitive_config(self):
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}, 'identity': {'driver': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
        res = PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id'])
        self.assertEqual(config, res)
        PROVIDERS.domain_config_api.delete_config_options(self.domain['id'])
        self.assertDictEqual(res, PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id']))
        PROVIDERS.domain_config_api.get_config_with_sensitive_info.invalidate(PROVIDERS.domain_config_api, self.domain['id'])
        self.assertDictEqual({}, PROVIDERS.domain_config_api.get_config_with_sensitive_info(self.domain['id']))

    def test_delete_domain_deletes_configs(self):
        """Test domain deletion clears the domain configs."""
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        config = {'ldap': {'url': uuid.uuid4().hex, 'user_tree_dn': uuid.uuid4().hex, 'password': uuid.uuid4().hex}}
        PROVIDERS.domain_config_api.create_config(domain['id'], config)
        domain['enabled'] = False
        PROVIDERS.resource_api.update_domain(domain['id'], domain)
        PROVIDERS.resource_api.delete_domain(domain['id'])
        self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, domain['id'])
        self.assertDictEqual({}, PROVIDERS.domain_config_api.get_config_with_sensitive_info(domain['id']))

    def test_config_registration(self):
        type = uuid.uuid4().hex
        PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type)
        PROVIDERS.domain_config_api.release_registration(self.domain['id'], type=type)
        PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type)
        self.assertFalse(PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type))
        self.assertEqual(self.domain['id'], PROVIDERS.domain_config_api.read_registration(type))
        domain2 = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        PROVIDERS.domain_config_api.release_registration(domain2['id'], type=type)
        PROVIDERS.domain_config_api.release_registration(self.domain['id'], type=type)
        self.assertRaises(exception.ConfigRegistrationNotFound, PROVIDERS.domain_config_api.read_registration, type)
        type2 = uuid.uuid4().hex
        PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type)
        PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type2)
        PROVIDERS.domain_config_api.release_registration(self.domain['id'])
        self.assertRaises(exception.ConfigRegistrationNotFound, PROVIDERS.domain_config_api.read_registration, type)
        self.assertRaises(exception.ConfigRegistrationNotFound, PROVIDERS.domain_config_api.read_registration, type2)

    def test_option_dict_fails_when_group_is_none(self):
        group = 'foo'
        option = 'bar'
        self.assertRaises(cfg.NoSuchOptError, PROVIDERS.domain_config_api._option_dict, group, option)

    def test_option_dict_returns_valid_config_values(self):
        regex = uuid.uuid4().hex
        self.config_fixture.config(group='security_compliance', password_regex=regex)
        expected_dict = {'group': 'security_compliance', 'option': 'password_regex', 'value': regex}
        option_dict = PROVIDERS.domain_config_api._option_dict('security_compliance', 'password_regex')
        self.assertEqual(option_dict, expected_dict)