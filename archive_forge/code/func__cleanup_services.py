import random
import string
from openstack.cloud.exc import OpenStackCloudUnavailableFeature
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_services(self):
    exception_list = list()
    for s in self.operator_cloud.list_services():
        if s['name'] is not None and s['name'].startswith(self.new_item_name):
            try:
                self.operator_cloud.delete_service(name_or_id=s['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))