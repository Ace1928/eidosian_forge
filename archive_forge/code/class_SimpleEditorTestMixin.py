import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
class SimpleEditorTestMixin:

    def setUp(self):
        import traits.editor_factories
        self.factory = getattr(traits.editor_factories, self.factory_name)
        self.traitsui_factory = getattr(traitsui.api, self.traitsui_name)

    def test_editor(self):
        editor = self.factory()
        if isinstance(self.traitsui_factory, traitsui.api.BasicEditorFactory):
            self.assertIsInstance(editor, traitsui.api.BasicEditorFactory)
        else:
            self.assertIsInstance(editor, self.traitsui_factory)