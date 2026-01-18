from .. import config, debug, tests
class TestDebugFlags(tests.TestCaseInTempDir):

    def test_set_no_debug_flags_from_config(self):
        self.assertDebugFlags([], b'')

    def test_set_single_debug_flags_from_config(self):
        self.assertDebugFlags(['hpss'], b'debug_flags = hpss\n')

    def test_set_multiple_debug_flags_from_config(self):
        self.assertDebugFlags(['hpss', 'error'], b'debug_flags = hpss, error\n')

    def assertDebugFlags(self, expected_flags, conf_bytes):
        conf = config.GlobalStack()
        conf.store._load_from_string(b'[DEFAULT]\n' + conf_bytes)
        conf.store.save()
        self.overrideAttr(debug, 'debug_flags', set())
        debug.set_debug_flags_from_config()
        self.assertEqual(set(expected_flags), debug.debug_flags)