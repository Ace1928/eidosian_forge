import os
from pathlib import Path
import unittest
from traits.api import File, HasTraits, TraitError
from traits.testing.optional_dependencies import requires_traitsui
class TestCreateEditor(unittest.TestCase):

    @requires_traitsui
    def test_exists_controls_editor_dialog_style(self):
        x = File(exists=True)
        editor = x.create_editor()
        self.assertEqual(editor.dialog_style, 'open')
        x = File(exists=False)
        editor = x.create_editor()
        self.assertEqual(editor.dialog_style, 'save')