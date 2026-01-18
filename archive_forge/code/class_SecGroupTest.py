import testtools
from unittest import mock
from troveclient.v1 import security_groups
class SecGroupTest(testtools.TestCase):

    def setUp(self):
        super(SecGroupTest, self).setUp()
        self.orig__init = security_groups.SecurityGroup.__init__
        security_groups.SecurityGroup.__init__ = mock.Mock(return_value=None)
        self.security_group = security_groups.SecurityGroup()
        self.security_groups = security_groups.SecurityGroups(1)

    def tearDown(self):
        super(SecGroupTest, self).tearDown()
        security_groups.SecurityGroup.__init__ = self.orig__init

    def test___repr__(self):
        self.security_group.name = 'security_group-1'
        self.assertEqual('<SecurityGroup: security_group-1>', self.security_group.__repr__())

    def test_list(self):
        sec_group_list = ['secgroup1', 'secgroup2']
        self.security_groups.list = mock.Mock(return_value=sec_group_list)
        self.assertEqual(sec_group_list, self.security_groups.list())

    def test_get(self):

        def side_effect_func(path, inst):
            return (path, inst)
        self.security_groups._get = mock.Mock(side_effect=side_effect_func)
        self.security_group.id = 1
        self.assertEqual(('/security-groups/1', 'security_group'), self.security_groups.get(self.security_group))