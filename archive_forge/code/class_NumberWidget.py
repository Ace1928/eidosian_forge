import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
class NumberWidget(Widget):
    f = Float().tag(sync=True)
    cf = CFloat().tag(sync=True)
    i = Int().tag(sync=True)
    ci = CInt().tag(sync=True)