from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class OldControlComponentFormat(controldir.ControlComponentFormat):

    def get_format_description(self):
        return 'An old format that is slow'
    upgrade_recommended = True