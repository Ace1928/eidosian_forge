from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class TestProber(tests.TestCaseWithTransport):
    """Per-prober tests."""
    scenarios = [(prober_cls.__name__, {'prober_cls': prober_cls}) for prober_cls in controldir.ControlDirFormat._probers]

    def setUp(self):
        super().setUp()
        self.prober = self.prober_cls()

    def test_priority(self):
        transport = self.get_transport('.')
        self.assertIsInstance(self.prober.priority(transport), int)

    def test_probe_transport_empty(self):
        transport = self.get_transport('.')
        self.assertRaises(errors.NotBranchError, self.prober.probe_transport, transport)

    def test_known_formats(self):
        known_formats = self.prober_cls.known_formats()
        self.assertIsInstance(known_formats, list)
        for format in known_formats:
            self.assertIsInstance(format, controldir.ControlDirFormat, repr(format))