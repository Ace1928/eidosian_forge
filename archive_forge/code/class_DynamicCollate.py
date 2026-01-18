import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
class DynamicCollate(LoggingComparisonTestCase):

    def test_dynamic_collate_layout(self):

        def callback():
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        dmap = DynamicMap(callback, kdims=[])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [('Image', 'I'), ('Text', 'I')])
        self.assertEqual(layout.Image.I[()], Image(np.array([[0, 1], [2, 3]])))

    def test_dynamic_collate_layout_raise_no_remapping_error(self):

        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        stream = PointerXY()
        cb_callable = Callable(callback)
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        with self.assertRaisesRegex(ValueError, 'The following streams are set to be automatically linked'):
            dmap.collate()

    def test_dynamic_collate_layout_raise_ambiguous_remapping_error(self):

        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Image(np.array([[0, 1], [2, 3]]))
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={'Image': [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        with self.assertRaisesRegex(ValueError, 'The stream_mapping supplied on the Callable is ambiguous'):
            dmap.collate()

    def test_dynamic_collate_layout_with_integer_stream_mapping(self):

        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={0: [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [('Image', 'I'), ('Text', 'I')])
        self.assertIs(stream.source, layout.Image.I)

    def test_dynamic_collate_layout_with_spec_stream_mapping(self):

        def callback(x, y):
            return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={'Image': [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [('Image', 'I'), ('Text', 'I')])
        self.assertIs(stream.source, layout.Image.I)

    def test_dynamic_collate_ndlayout(self):

        def callback():
            return NdLayout({i: Image(np.array([[i, 1], [2, 3]])) for i in range(1, 3)})
        dmap = DynamicMap(callback, kdims=[])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [1, 2])
        self.assertEqual(layout[1][()], Image(np.array([[1, 1], [2, 3]])))

    def test_dynamic_collate_ndlayout_with_integer_stream_mapping(self):

        def callback(x, y):
            return NdLayout({i: Image(np.array([[i, 1], [2, 3]])) for i in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={0: [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [1, 2])
        self.assertIs(stream.source, layout[1])

    def test_dynamic_collate_ndlayout_with_key_stream_mapping(self):

        def callback(x, y):
            return NdLayout({i: Image(np.array([[i, 1], [2, 3]])) for i in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={(1,): [stream]})
        dmap = DynamicMap(cb_callable, kdims=[], streams=[stream])
        layout = dmap.collate()
        self.assertEqual(list(layout.keys()), [1, 2])
        self.assertIs(stream.source, layout[1])

    def test_dynamic_collate_grid(self):

        def callback():
            return GridSpace({(i, j): Image(np.array([[i, j], [2, 3]])) for i in range(1, 3) for j in range(1, 3)})
        dmap = DynamicMap(callback, kdims=[])
        grid = dmap.collate()
        self.assertEqual(list(grid.keys()), [(i, j) for i in range(1, 3) for j in range(1, 3)])
        self.assertEqual(grid[0, 1][()], Image(np.array([[1, 1], [2, 3]])))

    def test_dynamic_collate_grid_with_integer_stream_mapping(self):

        def callback():
            return GridSpace({(i, j): Image(np.array([[i, j], [2, 3]])) for i in range(1, 3) for j in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={1: [stream]})
        dmap = DynamicMap(cb_callable, kdims=[])
        grid = dmap.collate()
        self.assertEqual(list(grid.keys()), [(i, j) for i in range(1, 3) for j in range(1, 3)])
        self.assertEqual(stream.source, grid[1, 2])

    def test_dynamic_collate_grid_with_key_stream_mapping(self):

        def callback():
            return GridSpace({(i, j): Image(np.array([[i, j], [2, 3]])) for i in range(1, 3) for j in range(1, 3)})
        stream = PointerXY()
        cb_callable = Callable(callback, stream_mapping={(1, 2): [stream]})
        dmap = DynamicMap(cb_callable, kdims=[])
        grid = dmap.collate()
        self.assertEqual(list(grid.keys()), [(i, j) for i in range(1, 3) for j in range(1, 3)])
        self.assertEqual(stream.source, grid[1, 2])

    def test_dynamic_collate_layout_with_changing_label(self):

        def callback(i):
            return Layout([Curve([], label=str(j)) for j in range(i, i + 2)])
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(0, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        el1, el2 = (dmap1[2], dmap2[2])
        self.assertEqual(el1.label, '2')
        self.assertEqual(el2.label, '3')

    def test_dynamic_collate_ndlayout_with_changing_keys(self):

        def callback(i):
            return NdLayout({j: Curve([], label=str(j)) for j in range(i, i + 2)})
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(0, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        el1, el2 = (dmap1[2], dmap2[2])
        self.assertEqual(el1.label, '2')
        self.assertEqual(el2.label, '3')

    def test_dynamic_collate_gridspace_with_changing_keys(self):

        def callback(i):
            return GridSpace({j: Curve([], label=str(j)) for j in range(i, i + 2)}, 'X')
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(0, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        el1, el2 = (dmap1[2], dmap2[2])
        self.assertEqual(el1.label, '2')
        self.assertEqual(el2.label, '3')

    def test_dynamic_collate_gridspace_with_changing_items_raises(self):

        def callback(i):
            return GridSpace({j: Curve([], label=str(j)) for j in range(i)}, 'X')
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(2, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        err = 'Collated DynamicMaps must return GridSpace with consistent number of items.'
        with self.assertRaisesRegex(ValueError, err):
            dmap1[4]
        self.log_handler.assertContains('WARNING', err)

    def test_dynamic_collate_gridspace_with_changing_item_types_raises(self):

        def callback(i):
            eltype = Image if i % 2 else Curve
            return GridSpace({j: eltype([], label=str(j)) for j in range(i, i + 2)}, 'X')
        dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(2, 10))
        layout = dmap.collate()
        dmap1, dmap2 = layout.values()
        err = 'The objects in a GridSpace returned by a DynamicMap must consistently return the same number of items of the same type.'
        with self.assertRaisesRegex(ValueError, err):
            dmap1[3]
        self.log_handler.assertContains('WARNING', err)