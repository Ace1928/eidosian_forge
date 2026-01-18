from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class SampleComponentFormat(controldir.ControlComponentFormat):

    def get_format_string(self):
        return b'Example component format.'