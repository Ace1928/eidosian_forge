import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def _x_test_set_state_int_to_int_like():
    w = NumberWidget()
    w.set_state(dict(i=3.0))
    assert len(w.comm.messages) == 0