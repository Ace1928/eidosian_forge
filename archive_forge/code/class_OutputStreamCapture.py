import gettext
import logging
import os
import warnings
import fixtures
from oslo_config import cfg
class OutputStreamCapture(fixtures.Fixture):
    """Capture output streams during tests.

    This fixture captures errant printing to stderr / stdout during
    the tests and lets us see those streams at the end of the test
    runs instead. Useful to see what was happening during failed
    tests.
    """

    def setUp(self):
        super(OutputStreamCapture, self).setUp()
        if os.environ.get('OS_STDOUT_CAPTURE') in _TRUE_VALUES:
            self.out = self.useFixture(fixtures.StringStream('stdout'))
            self.useFixture(fixtures.MonkeyPatch('sys.stdout', self.out.stream))
        if os.environ.get('OS_STDERR_CAPTURE') in _TRUE_VALUES:
            self.err = self.useFixture(fixtures.StringStream('stderr'))
            self.useFixture(fixtures.MonkeyPatch('sys.stderr', self.err.stream))

    @property
    def stderr(self):
        return self.err._details['stderr'].as_text()

    @property
    def stdout(self):
        return self.out._details['stdout'].as_text()