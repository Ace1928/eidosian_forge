import random
import string
from openstack.cloud.exc import OpenStackCloudUnavailableFeature
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_endpoints(self):
    exception_list = list()
    for e in self.operator_cloud.list_endpoints():
        if e.get('region') is not None and e['region'].startswith(self.new_item_name):
            try:
                self.operator_cloud.delete_endpoint(id=e['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))