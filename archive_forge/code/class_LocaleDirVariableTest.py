from oslotest import base as test_base
import testscenarios.testcase
from oslo_i18n import _locale
class LocaleDirVariableTest(testscenarios.testcase.WithScenarios, test_base.BaseTestCase):
    scenarios = [('simple', {'domain': 'simple', 'expected': 'SIMPLE_LOCALEDIR'}), ('with_dot', {'domain': 'one.two', 'expected': 'ONE_TWO_LOCALEDIR'}), ('with_dash', {'domain': 'one-two', 'expected': 'ONE_TWO_LOCALEDIR'})]

    def test_make_variable_name(self):
        var = _locale.get_locale_dir_variable_name(self.domain)
        self.assertEqual(self.expected, var)