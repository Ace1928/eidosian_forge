from neutron_lib.plugins.ml2 import api
from neutron_lib.tests import _base as base
class _MechanismDriver(api.MechanismDriver):

    def bind_port(s, c):
        return c

    def initialize(self):
        pass