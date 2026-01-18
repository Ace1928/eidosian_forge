import datetime
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def _create_role_assignments(self):
    try:
        user = self.openstack('configuration show -f value -c auth.username')
    except Exception:
        user = self.openstack('configuration show -f value -c auth.user_id')
    try:
        user_domain = self.openstack('configuration show -f value -c auth.user_domain_name')
    except Exception:
        user_domain = self.openstack('configuration show -f value -c auth.user_domain_id')
    try:
        project = self.openstack('configuration show -f value -c auth.project_name')
    except Exception:
        project = self.openstack('configuration show -f value -c auth.project_id')
    try:
        project_domain = self.openstack('configuration show -f value -c auth.project_domain_name')
    except Exception:
        project_domain = self.openstack('configuration show -f value -c auth.project_domain_id')
    role1 = self._create_dummy_role()
    role2 = self._create_dummy_role()
    for role in (role1, role2):
        self.openstack('role add --user %(user)s --user-domain %(user_domain)s --project %(project)s --project-domain %(project_domain)s %(role)s' % {'user': user, 'user_domain': user_domain, 'project': project, 'project_domain': project_domain, 'role': role})
        self.addCleanup(self.openstack, 'role remove --user %(user)s --user-domain %(user_domain)s --project %(project)s --project-domain %(project_domain)s %(role)s' % {'user': user, 'user_domain': user_domain, 'project': project, 'project_domain': project_domain, 'role': role})
    return (role1, role2)