from unittest import mock
from troveclient.tests import fakes
from troveclient.tests.osc import utils
from troveclient.v1 import backups
from troveclient.v1 import clusters
from troveclient.v1 import configurations
from troveclient.v1 import databases
from troveclient.v1 import datastores
from troveclient.v1 import flavors
from troveclient.v1 import instances
from troveclient.v1 import limits
from troveclient.v1 import modules
from troveclient.v1 import quota
from troveclient.v1 import users
class FakeConfigurations(object):
    fake_config = fakes.FakeHTTPClient().get_configurations()[2]['configurations']
    fake_config_instances = fakes.FakeHTTPClient().get_configurations_c_123_instances()[2]
    fake_default_config = fakes.FakeHTTPClient().get_instances_1234_configuration()[2]['instance']

    def get_configurations_c_123(self):
        return configurations.Configuration(None, self.fake_config[0])

    def get_configuration_instances(self):
        return [instances.Instance(None, fake_instance) for fake_instance in self.fake_config_instances['instances']]

    def get_default_configuration(self):
        return instances.Instance(None, self.fake_default_config)