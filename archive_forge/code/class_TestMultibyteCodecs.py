import sys
import unittest
from breezy import tests
class TestMultibyteCodecs(tests.TestCaseWithTransport):
    """Tests for quirks of multibyte encodings and their python codecs"""

    def test_plugins_mbcs(self):
        """Ensure the plugins command works with cjkcodecs, see lp:754082"""
        self.disable_missing_extensions_warning()
        out, err = self.run_bzr(['plugins'], encoding='EUC-JP')
        self.assertEqual('', err)