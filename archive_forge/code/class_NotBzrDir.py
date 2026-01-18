from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class NotBzrDir(controldir.ControlDir):
    """A non .bzr based control directory."""

    def __init__(self, transport, format):
        self._format = format
        self.root_transport = transport
        self.transport = transport.clone('.not')