import testtools
from openstack import exceptions
from openstack.tests.functional import base
def _assert_project(self, volume_name_or_id, project_id, allowed=True):
    acls = self.operator_cloud.get_volume_type_access(volume_name_or_id)
    allowed_projects = [x.get('project_id') for x in acls]
    self.assertEqual(allowed, project_id in allowed_projects)