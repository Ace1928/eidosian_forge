import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import datastores
class DatastoreVersionMembersTest(testtools.TestCase):

    def setUp(self):
        super(DatastoreVersionMembersTest, self).setUp()
        self.orig__init = datastores.DatastoreVersionMembers.__init__
        datastores.DatastoreVersionMembers.__init__ = mock.Mock(return_value=None)
        self.datastore_version_members = datastores.DatastoreVersionMembers()
        self.datastore_version_members.api = mock.Mock()
        self.datastore_version_members.api.client = mock.Mock()
        self.datastore_version_members.resource_class = mock.Mock(return_value='ds_version_member-1')
        self.orig_base_getid = base.getid
        base.getid = mock.Mock(return_value='datastore_version_member1')

    def tearDown(self):
        super(DatastoreVersionMembersTest, self).tearDown()
        datastores.DatastoreVersionMembers.__init__ = self.orig__init
        base.getid = self.orig_base_getid

    def test_add(self):

        def side_effect_func(path, body, inst):
            return (path, body, inst)
        self.datastore_version_members._create = mock.Mock(side_effect=side_effect_func)
        p, b, i = self.datastore_version_members.add('data_store1', 'datastore_version1', 'tenant1')
        self.assertEqual('/mgmt/datastores/data_store1/versions/datastore_version1/members', p)
        self.assertEqual('datastore_version_member', i)
        self.assertEqual('tenant1', b['member'])

    def test_delete(self):

        def side_effect_func(path):
            return path
        self.datastore_version_members._delete = mock.Mock(side_effect=side_effect_func)
        p = self.datastore_version_members.delete('data_store1', 'datastore_version1', 'tenant1')
        self.assertEqual('/mgmt/datastores/data_store1/versions/datastore_version1/members/tenant1', p)

    def test_list(self):
        page_mock = mock.Mock()
        self.datastore_version_members._list = page_mock
        limit = 'test-limit'
        marker = 'test-marker'
        self.datastore_version_members.list('datastore1', 'datastore_version1', limit, marker)
        page_mock.assert_called_with('/mgmt/datastores/datastore1/versions/datastore_version1/members', 'datastore_version_members', limit, marker)

    def test_get(self):

        def side_effect_func(path, inst):
            return (path, inst)
        self.datastore_version_members._get = mock.Mock(side_effect=side_effect_func)
        self.assertEqual(('/mgmt/datastores/datastore1/versions/datastore_version1/members/tenant1', 'datastore_version_member'), self.datastore_version_members.get('datastore1', 'datastore_version1', 'tenant1'))

    def test_get_by_tenant(self):
        page_mock = mock.Mock()
        self.datastore_version_members._list = page_mock
        limit = 'test-limit'
        marker = 'test-marker'
        self.datastore_version_members.get_by_tenant('datastore1', 'tenant1', limit, marker)
        page_mock.assert_called_with('/mgmt/datastores/datastore1/versions/members/tenant1', 'datastore_version_members', limit, marker)