from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import node
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
@mock.patch.object(exceptions, 'raise_from_response', mock.Mock())
class TestNodeInjectNMI(base.TestCase):

    def setUp(self):
        super().setUp()
        self.node = node.Node(**FAKE)
        self.session = mock.Mock(spec=adapter.Adapter)
        self.session.default_microversion = '1.29'
        self.node = node.Node(**FAKE)

    def test_inject_nmi(self):
        self.node.inject_nmi(self.session)
        self.session.put.assert_called_once_with('nodes/%s/management/inject_nmi' % FAKE['uuid'], json={}, headers=mock.ANY, microversion='1.29', retriable_status_codes=_common.RETRIABLE_STATUS_CODES)

    def test_incompatible_microversion(self):
        self.session.default_microversion = '1.28'
        self.assertRaises(exceptions.NotSupported, self.node.inject_nmi, self.session)