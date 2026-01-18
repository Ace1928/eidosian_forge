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
class DomainConfigDriverTests(object):

    def _domain_config_crud(self, sensitive):
        domain = uuid.uuid4().hex
        group = uuid.uuid4().hex
        option = uuid.uuid4().hex
        value = uuid.uuid4().hex
        config = {'group': group, 'option': option, 'value': value, 'sensitive': sensitive}
        self.driver.create_config_options(domain, [config])
        res = self.driver.get_config_option(domain, group, option, sensitive)
        config.pop('sensitive')
        self.assertEqual(config, res)
        value = uuid.uuid4().hex
        config = {'group': group, 'option': option, 'value': value, 'sensitive': sensitive}
        self.driver.update_config_options(domain, [config])
        res = self.driver.get_config_option(domain, group, option, sensitive)
        config.pop('sensitive')
        self.assertEqual(config, res)
        self.driver.delete_config_options(domain, group, option)
        self.assertRaises(exception.DomainConfigNotFound, self.driver.get_config_option, domain, group, option, sensitive)
        self.driver.delete_config_options(domain, group, option)

    def test_whitelisted_domain_config_crud(self):
        self._domain_config_crud(sensitive=False)

    def test_sensitive_domain_config_crud(self):
        self._domain_config_crud(sensitive=True)

    def _list_domain_config(self, sensitive):
        """Test listing by combination of domain, group & option."""
        config1 = {'group': uuid.uuid4().hex, 'option': uuid.uuid4().hex, 'value': uuid.uuid4().hex, 'sensitive': sensitive}
        config2 = {'group': config1['group'], 'option': uuid.uuid4().hex, 'value': uuid.uuid4().hex, 'sensitive': sensitive}
        config3 = {'group': uuid.uuid4().hex, 'option': uuid.uuid4().hex, 'value': 100, 'sensitive': sensitive}
        domain = uuid.uuid4().hex
        self.driver.create_config_options(domain, [config1, config2, config3])
        for config in [config1, config2, config3]:
            config.pop('sensitive')
        res = self.driver.list_config_options(domain, sensitive=sensitive)
        self.assertThat(res, matchers.HasLength(3))
        for res_entry in res:
            self.assertIn(res_entry, [config1, config2, config3])
        res = self.driver.list_config_options(domain, group=config1['group'], sensitive=sensitive)
        self.assertThat(res, matchers.HasLength(2))
        for res_entry in res:
            self.assertIn(res_entry, [config1, config2])
        res = self.driver.list_config_options(domain, group=config2['group'], option=config2['option'], sensitive=sensitive)
        self.assertThat(res, matchers.HasLength(1))
        self.assertEqual(config2, res[0])

    def test_list_whitelisted_domain_config_crud(self):
        self._list_domain_config(False)

    def test_list_sensitive_domain_config_crud(self):
        self._list_domain_config(True)

    def _delete_domain_configs(self, sensitive):
        """Test deleting by combination of domain, group & option."""
        config1 = {'group': uuid.uuid4().hex, 'option': uuid.uuid4().hex, 'value': uuid.uuid4().hex, 'sensitive': sensitive}
        config2 = {'group': config1['group'], 'option': uuid.uuid4().hex, 'value': uuid.uuid4().hex, 'sensitive': sensitive}
        config3 = {'group': config1['group'], 'option': uuid.uuid4().hex, 'value': uuid.uuid4().hex, 'sensitive': sensitive}
        config4 = {'group': uuid.uuid4().hex, 'option': uuid.uuid4().hex, 'value': uuid.uuid4().hex, 'sensitive': sensitive}
        domain = uuid.uuid4().hex
        self.driver.create_config_options(domain, [config1, config2, config3, config4])
        for config in [config1, config2, config3, config4]:
            config.pop('sensitive')
        res = self.driver.delete_config_options(domain, group=config2['group'], option=config2['option'])
        res = self.driver.list_config_options(domain, sensitive=sensitive)
        self.assertThat(res, matchers.HasLength(3))
        for res_entry in res:
            self.assertIn(res_entry, [config1, config3, config4])
        res = self.driver.delete_config_options(domain, group=config4['group'])
        res = self.driver.list_config_options(domain, sensitive=sensitive)
        self.assertThat(res, matchers.HasLength(2))
        for res_entry in res:
            self.assertIn(res_entry, [config1, config3])
        res = self.driver.delete_config_options(domain)
        res = self.driver.list_config_options(domain, sensitive=sensitive)
        self.assertThat(res, matchers.HasLength(0))

    def test_delete_whitelisted_domain_configs(self):
        self._delete_domain_configs(False)

    def test_delete_sensitive_domain_configs(self):
        self._delete_domain_configs(True)

    def _create_domain_config_twice(self, sensitive):
        """Test create the same option twice just overwrites."""
        config = {'group': uuid.uuid4().hex, 'option': uuid.uuid4().hex, 'value': uuid.uuid4().hex, 'sensitive': sensitive}
        domain = uuid.uuid4().hex
        self.driver.create_config_options(domain, [config])
        config['value'] = uuid.uuid4().hex
        self.driver.create_config_options(domain, [config])
        res = self.driver.get_config_option(domain, config['group'], config['option'], sensitive)
        config.pop('sensitive')
        self.assertEqual(config, res)

    def test_create_whitelisted_domain_config_twice(self):
        self._create_domain_config_twice(False)

    def test_create_sensitive_domain_config_twice(self):
        self._create_domain_config_twice(True)