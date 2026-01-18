import pprint
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_projects(self):
    exception_list = list()
    for p in self.operator_cloud.list_projects():
        if p['name'].startswith(self.new_project_name):
            try:
                self.operator_cloud.delete_project(p['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))