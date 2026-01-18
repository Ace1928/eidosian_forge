import abc
from neutron_lib.services import base
from neutron_lib.tests import _base as test_base
class ServicePluginStub(base.ServicePluginBase):

    def get_plugin_type(self):
        pass

    def get_plugin_description(self):
        pass