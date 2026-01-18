from neutron_lib.api.definitions import quota_check_limit
from neutron_lib.tests.unit.api.definitions import base
class QuotaCheckLimitDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = quota_check_limit