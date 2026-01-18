import random
import string
from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_users(self):
    exception_list = list()
    for user in self.operator_cloud.list_users():
        if user['name'].startswith(self.user_prefix):
            try:
                self.operator_cloud.delete_user(user['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))