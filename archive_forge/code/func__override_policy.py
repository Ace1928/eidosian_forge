import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import base as bp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
def _override_policy(self):
    with open(self.policy_file_name, 'w') as f:
        overridden_policies = {'identity:list_projects_for_endpoint': bp.SYSTEM_READER, 'identity:add_endpoint_to_project': bp.SYSTEM_ADMIN, 'identity:check_endpoint_in_project': bp.SYSTEM_READER, 'identity:list_endpoints_for_project': bp.SYSTEM_READER, 'identity:remove_endpoint_from_project': bp.SYSTEM_ADMIN}
        f.write(jsonutils.dumps(overridden_policies))