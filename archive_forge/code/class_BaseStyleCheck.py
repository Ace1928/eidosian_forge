import textwrap
import pycodestyle
from keystone.tests.hacking import checks
from keystone.tests import unit
from keystone.tests.unit.ksfixtures import hacking as hacking_fixtures
class BaseStyleCheck(unit.BaseTestCase):

    def setUp(self):
        super(BaseStyleCheck, self).setUp()
        self.code_ex = self.useFixture(self.get_fixture())
        self.addCleanup(delattr, self, 'code_ex')

    def get_checker(self):
        """Return the checker to be used for tests in this class."""
        raise NotImplementedError('subclasses must provide a real implementation')

    def get_fixture(self):
        return hacking_fixtures.HackingCode()

    def run_check(self, code):
        pycodestyle.register_check(self.get_checker())
        lines = textwrap.dedent(code).strip().splitlines(True)
        guide = pycodestyle.StyleGuide(select='K')
        checker = pycodestyle.Checker(lines=lines, options=guide.options)
        checker.check_all()
        checker.report._deferred_print.sort()
        return checker.report._deferred_print

    def assert_has_errors(self, code, expected_errors=None):
        actual_errors = [e[:3] for e in self.run_check(code)]
        self.assertCountEqual(expected_errors or [], actual_errors)