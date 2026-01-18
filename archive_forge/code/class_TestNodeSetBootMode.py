from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(utils, 'pick_microversion', lambda session, v: v)
@mock.patch.object(node.Node, 'fetch', lambda self, session: self)
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
class TestNodeSetBootMode(base.TestCase):

    def setUp(self):
        super(TestNodeSetBootMode, self).setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter, default_microversion='1.1')

    def test_node_set_boot_mode(self):
        self.node.set_boot_mode(self.session, 'uefi')
        self.session.put.assert_called_once_with('nodes/%s/states/boot_mode' % self.node.id, json={'target': 'uefi'}, headers=mock.ANY, microversion=mock.ANY, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_node_set_boot_mode_invalid_mode(self):
        self.assertRaises(ValueError, self.node.set_boot_mode, self.session, 'invalid-efi')