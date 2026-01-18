from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_policies(self):
    exception_list = list()
    for policy in self.operator_cloud.list_qos_policies():
        if policy['name'].startswith(self.policy_name):
            try:
                self.operator_cloud.delete_qos_policy(policy['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))