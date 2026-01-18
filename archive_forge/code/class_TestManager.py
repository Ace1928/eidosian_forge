import uuid
from keystone.common import manager
from keystone.common import provider_api
from keystone.tests import unit
class TestManager(manager.Manager):
    _provides_api = provides_api
    driver_namespace = '_TEST_NOTHING'

    def do_something(self):
        return provides_api