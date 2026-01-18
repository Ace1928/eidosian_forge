import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
@observe('value')
def _propagate_value(self, change):
    print('_propagate_value', change.new)
    if change.new == 42:
        self.value = 2
        self.other = 11