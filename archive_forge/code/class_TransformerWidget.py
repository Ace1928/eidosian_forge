import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
class TransformerWidget(Widget):
    d = List(Bool()).tag(sync=True, from_json=transform_fromjson)