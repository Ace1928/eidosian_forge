from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class SampleExtraComponentFormat(controldir.ControlComponentFormat):
    """Extra format, no format string."""