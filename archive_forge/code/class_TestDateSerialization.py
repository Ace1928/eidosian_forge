import array
import datetime as dt
import pytest
from unittest import TestCase
from traitlets import HasTraits, Int, TraitError
from traitlets.tests.test_traitlets import TraitTestBase
from ipywidgets import Color, NumberFormat
from ipywidgets.widgets.widget import _remove_buffers, _put_buffers
from ipywidgets.widgets.trait_types import date_serialization, TypedTuple
class TestDateSerialization(TestCase):

    def setUp(self):
        self.to_json = date_serialization['to_json']
        self.dummy_manager = None

    def test_serialize_none(self):
        self.assertIs(self.to_json(None, self.dummy_manager), None)

    def test_serialize_date(self):
        date = dt.date(1900, 2, 18)
        expected = {'year': 1900, 'month': 1, 'date': 18}
        self.assertEqual(self.to_json(date, self.dummy_manager), expected)