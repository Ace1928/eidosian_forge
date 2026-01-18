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
class TestNodeTraits(base.TestCase):

    def setUp(self):
        super(TestNodeTraits, self).setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter, default_microversion='1.37')
        self.session.log = mock.Mock()

    def test_node_add_trait(self):
        self.node.add_trait(self.session, 'CUSTOM_FAKE')
        self.session.put.assert_called_once_with('nodes/%s/traits/%s' % (self.node.id, 'CUSTOM_FAKE'), json=None, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_remove_trait(self):
        self.assertTrue(self.node.remove_trait(self.session, 'CUSTOM_FAKE'))
        self.session.delete.assert_called_once_with('nodes/%s/traits/%s' % (self.node.id, 'CUSTOM_FAKE'), headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_remove_trait_missing(self):
        self.session.delete.return_value.status_code = 400
        self.assertFalse(self.node.remove_trait(self.session, 'CUSTOM_MISSING'))
        self.session.delete.assert_called_once_with('nodes/%s/traits/%s' % (self.node.id, 'CUSTOM_MISSING'), headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_set_traits(self):
        traits = ['CUSTOM_FAKE', 'CUSTOM_REAL', 'CUSTOM_MISSING']
        self.node.set_traits(self.session, traits)
        self.session.put.assert_called_once_with('nodes/%s/traits' % self.node.id, json={'traits': ['CUSTOM_FAKE', 'CUSTOM_REAL', 'CUSTOM_MISSING']}, headers=mock.ANY, microversion='1.37', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)