import keystoneauth1.exceptions as kc_exception
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
def _delete_keypair(self):
    credential_id = self.data().get('credential_id')
    if not credential_id:
        return
    user_id = self._get_user_id()
    if user_id is None:
        return
    try:
        self.keystone().delete_stack_domain_user_keypair(user_id=user_id, project_id=self.stack.stack_user_project_id, credential_id=credential_id)
    except kc_exception.NotFound:
        pass
    except ValueError:
        self.keystone().delete_ec2_keypair(user_id=user_id, credential_id=credential_id)
    for data_key in ('access_key', 'secret_key', 'credential_id'):
        self.data_delete(data_key)