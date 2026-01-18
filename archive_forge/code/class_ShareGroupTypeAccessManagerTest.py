from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_type_access as type_access
@ddt.ddt
class ShareGroupTypeAccessManagerTest(utils.TestCase):

    def setUp(self):
        super(ShareGroupTypeAccessManagerTest, self).setUp()
        self.manager = type_access.ShareGroupTypeAccessManager(fake.FakeClient())

    def test_list(self):
        fake_share_group_type_access = fake.ShareGroupTypeAccess()
        mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_type_access]))
        result = self.manager.list(fake.ShareGroupType(), search_opts=None)
        self.assertEqual([fake_share_group_type_access], result)
        mock_list.assert_called_once_with(type_access.RESOURCE_PATH % fake.ShareGroupType.id, type_access.RESOURCE_NAME)

    def test_list_public(self):
        fake_share_group_type_access = fake.ShareGroupTypeAccess()
        mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_type_access]))
        fake_share_group_type = fake.ShareGroupType()
        fake_share_group_type.is_public = True
        result = self.manager.list(fake_share_group_type)
        self.assertIsNone(result)
        self.assertFalse(mock_list.called)

    def test_list_using_unsupported_microversion(self):
        fake_share_group_type_access = fake.ShareGroupTypeAccess()
        self.manager.api.api_version = manilaclient.API_MIN_VERSION
        self.assertRaises(exceptions.UnsupportedVersion, self.manager.list, fake_share_group_type_access)

    def test_add_project_access(self):
        mock_post = self.mock_object(self.manager.api.client, 'post')
        self.manager.add_project_access(fake.ShareGroupType(), 'fake_project')
        expected_body = {'addProjectAccess': {'project': 'fake_project'}}
        mock_post.assert_called_once_with(type_access.RESOURCE_PATH_ACTION % fake.ShareGroupType.id, body=expected_body)

    def test_remove_project_access(self):
        mock_post = self.mock_object(self.manager.api.client, 'post')
        self.manager.remove_project_access(fake.ShareGroupType(), 'fake_project')
        expected_body = {'removeProjectAccess': {'project': 'fake_project'}}
        mock_post.assert_called_once_with(type_access.RESOURCE_PATH_ACTION % fake.ShareGroupType.id, body=expected_body)