import array
import datetime as dt
import pytest
from unittest import TestCase
from traitlets import HasTraits, Int, TraitError
from traitlets.tests.test_traitlets import TraitTestBase
from ipywidgets import Color, NumberFormat
from ipywidgets.widgets.widget import _remove_buffers, _put_buffers
from ipywidgets.widgets.trait_types import date_serialization, TypedTuple
class TestBuffers(TestCase):

    def test_remove_and_put_buffers(self):
        mv1 = memoryview(b'test1')
        mv2 = memoryview(b'test2')
        state = {'plain': [0, 'text'], 'x': {'ar': mv1}, 'y': {'shape': (10, 10), 'data': mv1}, 'z': (mv1, mv2), 'top': mv1, 'deep': {'a': 1, 'b': [0, {'deeper': mv2}]}}
        plain = state['plain']
        x = state['x']
        y = state['y']
        y_shape = y['shape']
        state_before = state
        state, buffer_paths, buffers = _remove_buffers(state)
        self.assertIn('plain', state)
        self.assertIn('shape', state['y'])
        self.assertNotIn('ar', state['x'])
        self.assertEqual(state['x'], {})
        self.assertNotIn('data', state['y'])
        self.assertNotIn(mv1, state['z'])
        self.assertNotIn(mv1, state['z'])
        self.assertNotIn('top', state)
        self.assertIn('deep', state)
        self.assertIn('b', state['deep'])
        self.assertNotIn('deeper', state['deep']['b'][1])
        self.assertIsNot(state, state_before)
        self.assertIs(state['plain'], plain)
        self.assertIsNot(state['x'], x)
        self.assertIsNot(state['y'], y)
        self.assertIs(state['y']['shape'], y_shape)
        for path, buffer in [(['x', 'ar'], mv1), (['y', 'data'], mv1), (['z', 0], mv1), (['z', 1], mv2), (['top'], mv1), (['deep', 'b', 1, 'deeper'], mv2)]:
            self.assertIn(path, buffer_paths, '%r not in path' % path)
            index = buffer_paths.index(path)
            self.assertEqual(buffer, buffers[index])
        _put_buffers(state, buffer_paths, buffers)
        state_before['z'] = list(state_before['z'])
        self.assertEqual(state_before, state)