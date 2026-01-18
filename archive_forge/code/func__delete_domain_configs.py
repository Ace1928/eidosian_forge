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