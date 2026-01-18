import textwrap
import pycodestyle
from keystone.tests.hacking import checks
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import hacking as hacking_fixtures
class TestCheckForMutableDefaultArgs(BaseStyleCheck):

    def get_checker(self):
        return checks.CheckForMutableDefaultArgs

    def test(self):
        code = self.code_ex.mutable_default_args['code']
        errors = self.code_ex.mutable_default_args['expected_errors']
        self.assert_has_errors(code, expected_errors=errors)