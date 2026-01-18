import inspect
from unittest import TestCase
from traitlets import TraitError
from ipywidgets import Dropdown, SelectionSlider, Select
class TestDropdown(TestCase):

    def test_construction(self):
        Dropdown()

    def test_dict_mapping_options(self):
        d = Dropdown(options={'One': 1, 'Two': 2, 'Three': 3})
        assert d.get_state('_options_labels') == {'_options_labels': ('One', 'Two', 'Three')}

    def test_setting_options_from_list(self):
        d = Dropdown()
        assert d.options == ()
        d.options = ['One', 'Two', 'Three']
        assert d.get_state('_options_labels') == {'_options_labels': ('One', 'Two', 'Three')}

    def test_setting_options_from_list_tuples(self):
        d = Dropdown()
        assert d.options == ()
        d.options = [('One', 1), ('Two', 2), ('Three', 3)]
        assert d.get_state('_options_labels') == {'_options_labels': ('One', 'Two', 'Three')}
        d.value = 2
        assert d.get_state('index') == {'index': 1}

    def test_setting_options_from_dict(self):
        d = Dropdown()
        assert d.options == ()
        d.options = {'One': 1, 'Two': 2, 'Three': 3}
        assert d.get_state('_options_labels') == {'_options_labels': ('One', 'Two', 'Three')}