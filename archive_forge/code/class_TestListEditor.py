import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
@requires_traitsui
class TestListEditor(unittest.TestCase):

    def test_list_editor_default(self):
        trait = List(Str)
        editor = list_editor(trait, trait)
        self.assertIsInstance(editor, traitsui.api.ListEditor)
        self.assertEqual(editor.trait_handler, trait)
        self.assertEqual(editor.rows, 5)
        self.assertFalse(editor.use_notebook)
        self.assertEqual(editor.page_name, '')

    def test_list_editor_options(self):
        trait = List(Str, rows=10, use_notebook=True, page_name='page')
        editor = list_editor(trait, trait)
        self.assertIsInstance(editor, traitsui.api.ListEditor)
        self.assertEqual(editor.trait_handler, trait)
        self.assertEqual(editor.rows, 10)
        self.assertTrue(editor.use_notebook)
        self.assertEqual(editor.page_name, 'page')

    def test_list_editor_list_instance(self):
        trait = List(Instance(HasTraits))
        editor = list_editor(trait, trait)
        self.assertIsInstance(editor, traitsui.api.TableEditor)

    def test_list_editor_list_instance_row_factory(self):
        trait = List(Instance(HasTraits, kw={}))
        editor = trait.create_editor()
        self.assertIsInstance(editor, traitsui.api.TableEditor)
        if editor.row_factory is not None:
            self.assertTrue(callable(editor.row_factory))