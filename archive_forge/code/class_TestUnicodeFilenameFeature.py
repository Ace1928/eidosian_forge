import sys
from .. import plugin as _mod_plugin
from .. import symbol_versioning, tests
from . import features
class TestUnicodeFilenameFeature(tests.TestCase):

    def test_probe_passes(self):
        """UnicodeFilenameFeature._probe passes."""
        features.UnicodeFilenameFeature._probe()