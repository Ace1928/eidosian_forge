import textwrap
import pycodestyle
from keystone.tests.hacking import checks
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import hacking as hacking_fixtures
class TestDictConstructorWithSequenceCopy(BaseStyleCheck):

    def get_checker(self):
        return checks.dict_constructor_with_sequence_copy

    def test(self):
        code = self.code_ex.dict_constructor['code']
        errors = self.code_ex.dict_constructor['expected_errors']
        self.assert_has_errors(code, expected_errors=errors)