import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_subnets(self):
    exception_list = list()
    for subnet in self.operator_cloud.list_subnets():
        if subnet['name'].startswith(self.subnet_prefix):
            try:
                self.operator_cloud.delete_subnet(subnet['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))