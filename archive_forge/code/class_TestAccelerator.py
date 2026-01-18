import copy
import uuid
from openstack.tests.unit import base
class TestAccelerator(base.TestCase):

    def setUp(self):
        super(TestAccelerator, self).setUp()
        self.use_cyborg()

    def test_list_deployables(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'deployables']), json={'deployables': [DEP_DICT]})])
        dep_list = self.cloud.list_deployables()
        self.assertEqual(len(dep_list), 1)
        self.assertEqual(dep_list[0].id, DEP_DICT['uuid'])
        self.assertEqual(dep_list[0].name, DEP_DICT['name'])
        self.assertEqual(dep_list[0].parent_id, DEP_DICT['parent_id'])
        self.assertEqual(dep_list[0].root_id, DEP_DICT['root_id'])
        self.assertEqual(dep_list[0].num_accelerators, DEP_DICT['num_accelerators'])
        self.assertEqual(dep_list[0].device_id, DEP_DICT['device_id'])
        self.assert_calls()

    def test_list_devices(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'devices']), json={'devices': [DEV_DICT]})])
        dev_list = self.cloud.list_devices()
        self.assertEqual(len(dev_list), 1)
        self.assertEqual(dev_list[0].id, DEV_DICT['id'])
        self.assertEqual(dev_list[0].uuid, DEV_DICT['uuid'])
        self.assertEqual(dev_list[0].name, DEV_DICT['name'])
        self.assertEqual(dev_list[0].type, DEV_DICT['type'])
        self.assertEqual(dev_list[0].vendor, DEV_DICT['vendor'])
        self.assertEqual(dev_list[0].model, DEV_DICT['model'])
        self.assertEqual(dev_list[0].std_board_info, DEV_DICT['std_board_info'])
        self.assertEqual(dev_list[0].vendor_board_info, DEV_DICT['vendor_board_info'])
        self.assert_calls()

    def test_list_device_profiles(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'device_profiles']), json={'device_profiles': [DEV_PROF_DICT]})])
        dev_prof_list = self.cloud.list_device_profiles()
        self.assertEqual(len(dev_prof_list), 1)
        self.assertEqual(dev_prof_list[0].id, DEV_PROF_DICT['id'])
        self.assertEqual(dev_prof_list[0].uuid, DEV_PROF_DICT['uuid'])
        self.assertEqual(dev_prof_list[0].name, DEV_PROF_DICT['name'])
        self.assertEqual(dev_prof_list[0].groups, DEV_PROF_DICT['groups'])
        self.assert_calls()

    def test_create_device_profile(self):
        self.register_uris([dict(method='POST', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'device_profiles']), json=NEW_DEV_PROF_DICT)])
        attrs = {'name': NEW_DEV_PROF_DICT['name'], 'groups': NEW_DEV_PROF_DICT['groups']}
        self.assertTrue(self.cloud.create_device_profile(attrs))
        self.assert_calls()

    def test_delete_device_profile(self, filters=None):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'device_profiles', DEV_PROF_DICT['name']]), json={'device_profiles': [DEV_PROF_DICT]}), dict(method='DELETE', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'device_profiles', DEV_PROF_DICT['name']]), json=DEV_PROF_DICT)])
        self.assertTrue(self.cloud.delete_device_profile(DEV_PROF_DICT['name'], filters))
        self.assert_calls()

    def test_list_accelerator_requests(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests']), json={'arqs': [ARQ_DICT]})])
        arq_list = self.cloud.list_accelerator_requests()
        self.assertEqual(len(arq_list), 1)
        self.assertEqual(arq_list[0].uuid, ARQ_DICT['uuid'])
        self.assertEqual(arq_list[0].device_profile_name, ARQ_DICT['device_profile_name'])
        self.assertEqual(arq_list[0].device_profile_group_id, ARQ_DICT['device_profile_group_id'])
        self.assertEqual(arq_list[0].device_rp_uuid, ARQ_DICT['device_rp_uuid'])
        self.assertEqual(arq_list[0].instance_uuid, ARQ_DICT['instance_uuid'])
        self.assertEqual(arq_list[0].attach_handle_type, ARQ_DICT['attach_handle_type'])
        self.assertEqual(arq_list[0].attach_handle_info, ARQ_DICT['attach_handle_info'])
        self.assert_calls()

    def test_create_accelerator_request(self):
        self.register_uris([dict(method='POST', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests']), json=NEW_ARQ_DICT)])
        attrs = {'device_profile_name': NEW_ARQ_DICT['device_profile_name'], 'device_profile_group_id': NEW_ARQ_DICT['device_profile_group_id']}
        self.assertTrue(self.cloud.create_accelerator_request(attrs))
        self.assert_calls()

    def test_delete_accelerator_request(self, filters=None):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json={'accelerator_requests': [ARQ_DICT]}), dict(method='DELETE', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json=ARQ_DICT)])
        self.assertTrue(self.cloud.delete_accelerator_request(ARQ_DICT['uuid'], filters))
        self.assert_calls()

    def test_bind_accelerator_request(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json={'accelerator_requests': [ARQ_DICT]}), dict(method='PATCH', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json=ARQ_DICT)])
        properties = [{'path': '/hostname', 'value': ARQ_DICT['hostname'], 'op': 'add'}, {'path': '/instance_uuid', 'value': ARQ_DICT['instance_uuid'], 'op': 'add'}, {'path': '/device_rp_uuid', 'value': ARQ_DICT['device_rp_uuid'], 'op': 'add'}]
        self.assertTrue(self.cloud.bind_accelerator_request(ARQ_DICT['uuid'], properties))
        self.assert_calls()

    def test_unbind_accelerator_request(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json={'accelerator_requests': [ARQ_DICT]}), dict(method='PATCH', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests', ARQ_DICT['uuid']]), json=ARQ_DICT)])
        properties = [{'path': '/hostname', 'op': 'remove'}, {'path': '/instance_uuid', 'op': 'remove'}, {'path': '/device_rp_uuid', 'op': 'remove'}]
        self.assertTrue(self.cloud.unbind_accelerator_request(ARQ_DICT['uuid'], properties))
        self.assert_calls()