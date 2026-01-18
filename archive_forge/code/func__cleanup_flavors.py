from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_flavors(self):
    exception_list = list()
    if self.operator_cloud:
        for f in self.operator_cloud.list_flavors(get_extra=False):
            if f['name'].startswith(self.new_item_name):
                try:
                    self.operator_cloud.delete_flavor(f['id'])
                except Exception as e:
                    exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))