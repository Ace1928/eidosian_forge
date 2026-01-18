import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
class HostMountManagerTestCase(base.BaseTestCase):

    class FakeHostMountState:

        def __init__(self):
            self.mountpoints = {mock.sentinel.mountpoint}

    def setUp(self):
        super(HostMountManagerTestCase, self).setUp()
        CONF.register_opt(cfg.DictOpt('enabled_backends'))
        CONF.set_override('enabled_backends', 'fake:file')
        if 'glance_store.common.fs_mount' in sys.modules:
            sys.modules.pop('glance_store.common.fs_mount')
        from glance_store.common import fs_mount as mount
        self.__manager__ = mount.__manager__

    def get_state(self):
        with self.__manager__.get_state() as state:
            return state

    def test_get_state_host_not_initialized(self):
        self.__manager__.state = None
        self.assertRaises(exceptions.HostNotInitialized, self.get_state)

    def test_get_state(self):
        self.__manager__.state = self.FakeHostMountState()
        state = self.get_state()
        self.assertEqual({mock.sentinel.mountpoint}, state.mountpoints)