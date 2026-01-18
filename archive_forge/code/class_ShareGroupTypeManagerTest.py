from unittest import mock
import ddt
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes as fake
from manilaclient.v2 import share_group_types as types
@ddt.ddt
class ShareGroupTypeManagerTest(utils.TestCase):

    def setUp(self):
        super(ShareGroupTypeManagerTest, self).setUp()
        self.manager = types.ShareGroupTypeManager(fake.FakeClient())
        self.fake_group_specs = {'key1': 'value1', 'key2': 'value2'}

    def test_create(self):
        fake_share_group_type = fake.ShareGroupType()
        mock_create = self.mock_object(self.manager, '_create', mock.Mock(return_value=fake_share_group_type))
        create_args = {'name': fake.ShareGroupType.name, 'share_types': [fake.ShareType()], 'is_public': False, 'group_specs': self.fake_group_specs}
        result = self.manager.create(**create_args)
        self.assertIs(fake_share_group_type, result)
        expected_body = {types.RESOURCE_NAME: {'name': fake.ShareGroupType.name, 'share_types': [fake.ShareType().id], 'is_public': False, 'group_specs': self.fake_group_specs}}
        mock_create.assert_called_once_with(types.RESOURCES_PATH, expected_body, types.RESOURCE_NAME)

    def test_create_no_share_type(self):
        create_args = {'name': fake.ShareGroupType.name, 'share_types': [], 'is_public': False, 'group_specs': self.fake_group_specs}
        self.assertRaises(ValueError, self.manager.create, **create_args)

    def test_create_using_unsupported_microversion(self):
        self.manager.api.api_version = manilaclient.API_MIN_VERSION
        self.assertRaises(exceptions.UnsupportedVersion, self.manager.create)

    def test_get(self):
        fake_share_group_type = fake.ShareGroupType()
        mock_get = self.mock_object(self.manager, '_get', mock.Mock(return_value=fake_share_group_type))
        result = self.manager.get(fake.ShareGroupType.id)
        self.assertIs(fake_share_group_type, result)
        mock_get.assert_called_once_with(types.RESOURCE_PATH % fake.ShareGroupType.id, types.RESOURCE_NAME)

    def test_list(self):
        fake_share_group_type = fake.ShareGroupType()
        mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_type]))
        result = self.manager.list(search_opts=None)
        self.assertEqual([fake_share_group_type], result)
        mock_list.assert_called_once_with(types.RESOURCES_PATH + '?is_public=all', types.RESOURCES_NAME)

    def test_list_no_public(self):
        fake_share_group_type = fake.ShareGroupType()
        mock_list = self.mock_object(self.manager, '_list', mock.Mock(return_value=[fake_share_group_type]))
        result = self.manager.list(show_all=False)
        self.assertEqual([fake_share_group_type], result)
        mock_list.assert_called_once_with(types.RESOURCES_PATH, types.RESOURCES_NAME)

    def test_delete(self):
        mock_delete = self.mock_object(self.manager, '_delete')
        self.manager.delete(fake.ShareGroupType())
        mock_delete.assert_called_once_with(types.RESOURCE_PATH % fake.ShareGroupType.id)