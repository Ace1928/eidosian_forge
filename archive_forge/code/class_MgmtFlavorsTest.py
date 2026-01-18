import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
class MgmtFlavorsTest(testtools.TestCase):

    def setUp(self):
        super(MgmtFlavorsTest, self).setUp()
        self.orig__init = management.MgmtFlavors.__init__
        management.MgmtFlavors.__init__ = mock.Mock(return_value=None)
        self.flavors = management.MgmtFlavors()
        self.flavors.api = mock.Mock()
        self.flavors.api.client = mock.Mock()
        self.flavors.resource_class = mock.Mock(return_value='flavor-1')
        self.orig_base_getid = base.getid
        base.getid = mock.Mock(return_value='flavor1')

    def tearDown(self):
        super(MgmtFlavorsTest, self).tearDown()
        management.MgmtFlavors.__init__ = self.orig__init
        base.getid = self.orig_base_getid

    def test_create(self):

        def side_effect_func(path, body, inst):
            return (path, body, inst)
        self.flavors._create = mock.Mock(side_effect=side_effect_func)
        p, b, i = self.flavors.create('test-name', 1024, 30, 2, 1)
        self.assertEqual('/mgmt/flavors', p)
        self.assertEqual('flavor', i)
        self.assertEqual('test-name', b['flavor']['name'])
        self.assertEqual(1024, b['flavor']['ram'])
        self.assertEqual(2, b['flavor']['vcpu'])
        self.assertEqual(1, b['flavor']['flavor_id'])