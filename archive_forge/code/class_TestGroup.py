import copy
from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import group
from openstack.tests.unit import base
class TestGroup(base.TestCase):

    def test_basic(self):
        resource = group.Group()
        self.assertEqual('group', resource.resource_key)
        self.assertEqual('groups', resource.resources_key)
        self.assertEqual('/groups', resource.base_path)
        self.assertTrue(resource.allow_create)
        self.assertTrue(resource.allow_fetch)
        self.assertTrue(resource.allow_delete)
        self.assertTrue(resource.allow_commit)
        self.assertTrue(resource.allow_list)

    def test_make_resource(self):
        resource = group.Group(**GROUP)
        self.assertEqual(GROUP['id'], resource.id)
        self.assertEqual(GROUP['status'], resource.status)
        self.assertEqual(GROUP['availability_zone'], resource.availability_zone)
        self.assertEqual(GROUP['created_at'], resource.created_at)
        self.assertEqual(GROUP['name'], resource.name)
        self.assertEqual(GROUP['description'], resource.description)
        self.assertEqual(GROUP['group_type'], resource.group_type)
        self.assertEqual(GROUP['volume_types'], resource.volume_types)
        self.assertEqual(GROUP['volumes'], resource.volumes)
        self.assertEqual(GROUP['group_snapshot_id'], resource.group_snapshot_id)
        self.assertEqual(GROUP['source_group_id'], resource.source_group_id)
        self.assertEqual(GROUP['project_id'], resource.project_id)