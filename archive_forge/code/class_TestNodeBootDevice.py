from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(node.Node, 'fetch', lambda self, session: self)
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
class TestNodeBootDevice(base.TestCase):

    def setUp(self):
        super().setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter, default_microversion='1.1')

    def test_get_boot_device(self):
        self.node.get_boot_device(self.session)
        self.session.get.assert_called_once_with('nodes/%s/management/boot_device' % self.node.id, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_set_boot_device(self):
        self.node.set_boot_device(self.session, 'pxe', persistent=False)
        self.session.put.assert_called_once_with('nodes/%s/management/boot_device' % self.node.id, json={'boot_device': 'pxe', 'persistent': False}, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_get_supported_boot_devices(self):
        self.node.get_supported_boot_devices(self.session)
        self.session.get.assert_called_once_with('nodes/%s/management/boot_device/supported' % self.node.id, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)