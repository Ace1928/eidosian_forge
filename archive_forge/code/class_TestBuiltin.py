from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import os_service_types
from os_service_types.tests import base
class TestBuiltin(base.TestCase, base.ServiceDataMixin):

    def setUp(self):
        super(TestBuiltin, self).setUp()
        self.service_types = os_service_types.ServiceTypes()

    def test_builtin_version(self):
        self.assertEqual(self.builtin_version, self.service_types.version)