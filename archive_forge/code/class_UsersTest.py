import testtools
from unittest import mock
from troveclient.v1 import users
class UsersTest(testtools.TestCase):

    def setUp(self):
        super(UsersTest, self).setUp()
        self.orig__init = users.Users.__init__
        users.Users.__init__ = mock.Mock(return_value=None)
        self.users = users.Users()
        self.users.api = mock.Mock()
        self.users.api.client = mock.Mock()
        self.instance_with_id = mock.Mock()
        self.instance_with_id.id = 215

    def tearDown(self):
        super(UsersTest, self).tearDown()
        users.Users.__init__ = self.orig__init

    def _get_mock_method(self):
        self._resp = mock.Mock()
        self._body = None
        self._url = None

        def side_effect_func(url, body=None):
            self._body = body
            self._url = url
            return (self._resp, body)
        return mock.Mock(side_effect=side_effect_func)

    def _build_fake_user(self, name, hostname=None, password=None, databases=None):
        return {'name': name, 'password': password if password else 'password', 'host': hostname, 'databases': databases if databases else []}

    def test_create(self):
        self.users.api.client.post = self._get_mock_method()
        self._resp.status_code = 200
        user = self._build_fake_user('user1')
        self.users.create(23, [user])
        self.assertEqual('/instances/23/users', self._url)
        self.assertEqual({'users': [user]}, self._body)
        del user['host']
        self.users.create(23, [user])
        self.assertEqual('/instances/23/users', self._url)
        user['host'] = '%'
        self.assertEqual({'users': [user]}, self._body)
        user['host'] = '127.0.0.1'
        self.users.create(23, [user])
        self.assertEqual({'users': [user]}, self._body)
        self.users.create(self.instance_with_id, [user])
        self.assertEqual({'users': [user]}, self._body)
        user['host'] = '%'
        self._resp.status_code = 400
        self.assertRaises(Exception, self.users.create, 12, [user])

    def test_delete(self):
        self.users.api.client.delete = self._get_mock_method()
        self._resp.status_code = 200
        self.users.delete(27, 'user1')
        self.assertEqual('/instances/27/users/user1', self._url)
        self.users.delete(self.instance_with_id, 'user1')
        self.assertEqual('/instances/%s/users/user1' % self.instance_with_id.id, self._url)
        self._resp.status_code = 400
        self.assertRaises(Exception, self.users.delete, 34, 'user1')

    def test_list(self):
        page_mock = mock.Mock()
        self.users._paginated = page_mock
        self.users.list('instance1')
        page_mock.assert_called_with('/instances/instance1/users', 'users', None, None)
        limit = 'test-limit'
        marker = 'test-marker'
        self.users.list('instance1', limit, marker)
        page_mock.assert_called_with('/instances/instance1/users', 'users', limit, marker)

    def test_update_no_changes(self):
        self.users.api.client.post = self._get_mock_method()
        self._resp.status_code = 200
        username = 'upd_user'
        user = self._build_fake_user(username)
        self.users.create(99, [user])
        instance = 'instance1'
        newuserattr = None
        self.assertRaises(Exception, self.users.update_attributes, instance, username, newuserattr)