import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def _override_policy_old_defaults(self):
    with open(self.policy_file_name, 'w') as f:
        overridden_policies = {'identity:list_trusts': '', 'identity:delete_trust': '', 'identity:get_trust': '', 'identity:list_roles_for_trust': '', 'identity:get_role_for_trust': ''}
        f.write(jsonutils.dumps(overridden_policies))