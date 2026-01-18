from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _process_app_cred(self, app_cred_ref):
    app_cred_ref = app_cred_ref.copy()
    app_cred_ref.pop('secret_hash')
    app_cred_ref['roles'] = self._get_role_list(app_cred_ref['roles'])
    return app_cred_ref