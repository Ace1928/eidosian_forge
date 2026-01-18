from unittest import mock
from keystoneauth1 import adapter
from openstack.identity.v3 import group
from openstack.identity.v3 import project
from openstack.identity.v3 import role
from openstack.identity.v3 import user
from openstack.tests.unit import base
class TestUserProject(base.TestCase):

    def test_basic(self):
        sot = project.UserProject()
        self.assertEqual('project', sot.resource_key)
        self.assertEqual('projects', sot.resources_key)
        self.assertEqual('/users/%(user_id)s/projects', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertFalse(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)