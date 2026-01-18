import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
class CannedInputUIFactoryTests(tests.TestCase):

    def test_canned_input_get_input(self):
        uif = _mod_ui.CannedInputUIFactory([True, 'mbp', 'password', 42])
        self.assertEqual(True, uif.get_boolean('Extra cheese?'))
        self.assertEqual('mbp', uif.get_username('Enter your user name'))
        self.assertEqual('password', uif.get_password('Password for %(host)s', host='example.com'))
        self.assertEqual(42, uif.get_integer('And all that jazz ?'))