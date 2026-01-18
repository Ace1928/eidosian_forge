from oslo_utils import reflection
from heat_integrationtests.common import test
class FunctionalTestsBase(test.HeatIntegrationTest):

    def setUp(self):
        super(FunctionalTestsBase, self).setUp()
        self.check_skip()

    def check_skip(self):
        test_cls_name = reflection.get_class_name(self, fully_qualified=False)
        test_method_name = '.'.join([test_cls_name, self._testMethodName])
        test_skipped = self.conf.skip_functional_test_list and (test_cls_name in self.conf.skip_functional_test_list or test_method_name in self.conf.skip_functional_test_list)
        if self.conf.skip_functional_tests or test_skipped:
            self.skipTest('Test disabled in conf, skipping')