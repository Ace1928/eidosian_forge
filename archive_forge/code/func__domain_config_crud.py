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