import inspect
from unittest import TestCase
from traitlets import TraitError
from ipywidgets import Dropdown, SelectionSlider, Select
class TestSelectionSlider(TestCase):

    def test_construction(self):
        SelectionSlider(options=['a', 'b', 'c'])

    def test_index_trigger(self):
        slider = SelectionSlider(options=['a', 'b', 'c'])
        observations = []

        def f(change):
            observations.append(change.new)
        slider.observe(f, 'index')
        assert slider.index == 0
        slider.options = [4, 5, 6]
        assert slider.index == 0
        assert slider.value == 4
        assert slider.label == '4'
        assert observations == [0]