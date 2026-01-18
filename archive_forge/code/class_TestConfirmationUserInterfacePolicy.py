import time
from testtools.matchers import *
from .. import config, tests
from .. import ui as _mod_ui
from ..bzr import remote
from ..ui import text as _mod_ui_text
from . import fixtures, ui_testing
from .testui import ProgressRecordingUIFactory
class TestConfirmationUserInterfacePolicy(tests.TestCase):

    def test_confirm_action_default(self):
        base_ui = _mod_ui.NoninteractiveUIFactory()
        for answer in [True, False]:
            self.assertEqual(_mod_ui.ConfirmationUserInterfacePolicy(base_ui, answer, {}).confirm_action('Do something?', 'breezy.tests.do_something', {}), answer)

    def test_confirm_action_specific(self):
        base_ui = _mod_ui.NoninteractiveUIFactory()
        for default_answer in [True, False]:
            for specific_answer in [True, False]:
                for conf_id in ['given_id', 'other_id']:
                    wrapper = _mod_ui.ConfirmationUserInterfacePolicy(base_ui, default_answer, dict(given_id=specific_answer))
                    result = wrapper.confirm_action('Do something?', conf_id, {})
                    if conf_id == 'given_id':
                        self.assertEqual(result, specific_answer)
                    else:
                        self.assertEqual(result, default_answer)

    def test_repr(self):
        base_ui = _mod_ui.NoninteractiveUIFactory()
        wrapper = _mod_ui.ConfirmationUserInterfacePolicy(base_ui, True, dict(a=2))
        self.assertThat(repr(wrapper), Equals("ConfirmationUserInterfacePolicy(NoninteractiveUIFactory(), True, {'a': 2})"))