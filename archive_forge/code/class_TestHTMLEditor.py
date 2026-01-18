import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
@requires_traitsui
class TestHTMLEditor(SimpleEditorTestMixin, unittest.TestCase):
    traitsui_name = 'HTMLEditor'
    factory_name = 'html_editor'