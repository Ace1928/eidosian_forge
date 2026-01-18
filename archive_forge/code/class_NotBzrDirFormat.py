from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class NotBzrDirFormat(controldir.ControlDirFormat):
    """A test class representing any non-.bzr based disk format."""

    def initialize_on_transport(self, transport):
        """Initialize a new .not dir in the base directory of a Transport."""
        transport.mkdir('.not')
        return self.open(transport)

    def open(self, transport):
        """Open this directory."""
        return NotBzrDir(transport, self)