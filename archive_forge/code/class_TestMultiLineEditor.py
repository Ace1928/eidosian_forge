import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
@requires_traitsui
class TestMultiLineEditor(unittest.TestCase):

    def test_multi_line_text_editor_default(self):
        editor = multi_line_text_editor()
        self.assertIsInstance(editor, traitsui.api.TextEditor)
        self.assertTrue(editor.multi_line)
        self.assertTrue(editor.auto_set)
        self.assertFalse(editor.enter_set)

    def test_multi_line_text_editor_options(self):
        editor = multi_line_text_editor(auto_set=False, enter_set=True)
        self.assertIsInstance(editor, traitsui.api.TextEditor)
        self.assertTrue(editor.multi_line)
        self.assertFalse(editor.auto_set)
        self.assertTrue(editor.enter_set)