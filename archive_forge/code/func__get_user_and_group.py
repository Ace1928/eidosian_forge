from openstack.cloud import _utils
from openstack import exceptions
from openstack.identity.v3._proxy import Proxy
from openstack import utils
def _get_user_and_group(self, user_name_or_id, group_name_or_id):
    user = self.get_user(user_name_or_id)
    if not user:
        raise exceptions.SDKException('User {user} not found'.format(user=user_name_or_id))
    group = self.get_group(group_name_or_id)
    if not group:
        raise exceptions.SDKException('Group {user} not found'.format(user=group_name_or_id))
    return (user, group)