import array
import datetime as dt
import pytest
from unittest import TestCase
from traitlets import HasTraits, Int, TraitError
from traitlets.tests.test_traitlets import TraitTestBase
from ipywidgets import Color, NumberFormat
from ipywidgets.widgets.widget import _remove_buffers, _put_buffers
from ipywidgets.widgets.trait_types import date_serialization, TypedTuple
class TestColor(TraitTestBase):
    obj = ColorTrait()
    _good_values = ['blue', '#AA0', '#FFFFFF', 'transparent', '#aaaa', '#ffffffff', 'rgb(0, 0, 0)', 'rgb( 20,70,50 )', 'rgba(10,10,10, 0.5)', 'rgba(255, 255, 255, 255)', 'hsl(0.0, .0, 0)', 'hsl( 0.5,0.3,0 )', 'hsla(10,10,10, 0.5)', 'var(--my-color)', 'var(--my-color-with_separators)', 'var(--my-color,)', 'var(--my-color-æ)', 'var(--my-color-ሴ)', 'var(--my-color-\\\\1234)', 'var(--my-color-\\.)', 'var(--my-color,black)', 'var(--my-color, black)', 'var(--my-color, rgb(20, 70, 50))', 'var(--my-color, #fff)']
    _bad_values = ['vanilla', 'blues', 1.2, 0.0, 0, 1, 2, 'rgb(0.4, 512, -40)', 'hsl(0.4, 512, -40)', 'rgba(0, 0, 0)', 'hsla(0, 0, 0)', 'var(-my-color)', 'var(--my-color-⁁)', 'var(my-color, black)', 'var(my-color-., black)', 'var(--my-color, vanilla)', 'var(--my-color, rgba(0,0,0))', None]