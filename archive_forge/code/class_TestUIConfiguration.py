import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
class TestUIConfiguration(tests.TestCaseInTempDir):

    def test_output_encoding_configuration(self):
        enc = next(fixtures.generate_unicode_encodings())
        config.GlobalStack().set('output_encoding', enc)
        IO = ui_testing.BytesIOWithEncoding
        ui = _mod_ui.make_ui_for_terminal(IO(), IO(), IO())
        output = ui.make_output_stream()
        self.assertEqual(output.encoding, enc)