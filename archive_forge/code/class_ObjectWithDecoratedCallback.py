from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
@registry.has_registry_receivers
class ObjectWithDecoratedCallback(object):

    def __init__(self):
        self.counter = 0

    @registry.receives(resources.PORT, [events.AFTER_CREATE, events.AFTER_UPDATE])
    @registry.receives(resources.NETWORK, [events.AFTER_DELETE])
    def callback(self, *args, **kwargs):
        self.counter += 1