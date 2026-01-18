from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
class TestLayoutProperties(TestCase):
    """test mixin with layout properties"""

    class DummyTemplate(widgets.GridBox, LayoutProperties):
        location = traitlets.Instance(widgets.Widget, allow_none=True)

    def test_layout_updated_on_trait_change(self):
        """test whether respective layout traits are updated when traits change"""
        template = self.DummyTemplate(width='100%')
        assert template.width == '100%'
        assert template.layout.width == '100%'
        template.width = 'auto'
        assert template.width == 'auto'
        assert template.layout.width == 'auto'

    def test_align_items_extra_options(self):
        template = self.DummyTemplate(align_items='top')
        assert template.align_items == 'top'
        assert template.layout.align_items == 'flex-start'
        template.align_items = 'bottom'
        assert template.align_items == 'bottom'
        assert template.layout.align_items == 'flex-end'

    def test_validate_properties(self):
        prop_obj = self.DummyTemplate()
        for prop in LayoutProperties.align_items.values:
            prop_obj.align_items = prop
            assert prop_obj.align_items == prop
        with pytest.raises(traitlets.TraitError):
            prop_obj.align_items = 'any default position'