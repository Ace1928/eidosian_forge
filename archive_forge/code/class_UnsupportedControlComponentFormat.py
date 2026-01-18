from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class UnsupportedControlComponentFormat(controldir.ControlComponentFormat):

    def is_supported(self):
        return False