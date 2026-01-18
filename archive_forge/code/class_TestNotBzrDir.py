from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class TestNotBzrDir(tests.TestCaseWithTransport):
    """Tests for using the controldir api with a non .bzr based disk format.

    If/when one of these is in the core, we can let the implementation tests
    verify this works.
    """

    def test_create_and_find_format(self):
        format = NotBzrDirFormat()
        dir = format.initialize(self.get_url())
        self.assertIsInstance(dir, NotBzrDir)
        controldir.ControlDirFormat.register_prober(NotBzrDirProber)
        try:
            found = controldir.ControlDirFormat.find_format(self.get_transport())
            self.assertIsInstance(found, NotBzrDirFormat)
        finally:
            controldir.ControlDirFormat.unregister_prober(NotBzrDirProber)

    def test_included_in_known_formats(self):
        controldir.ControlDirFormat.register_prober(NotBzrDirProber)
        self.addCleanup(controldir.ControlDirFormat.unregister_prober, NotBzrDirProber)
        formats = controldir.ControlDirFormat.known_formats()
        self.assertIsInstance(formats, list)
        for format in formats:
            if isinstance(format, NotBzrDirFormat):
                break
        else:
            self.fail('No NotBzrDirFormat in %s' % formats)