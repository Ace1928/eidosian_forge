from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
class TestDynamicMapInvocation(ComparisonTestCase):
    """
    Test that DynamicMap passes kdims and stream parameters correctly to
    Callables.
    """

    def test_dynamic_kdims_only(self):

        def fn(A, B):
            return Scatter([(B, 2)], label=A)
        dmap = DynamicMap(fn, kdims=['A', 'B'])
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_only_by_position(self):

        def fn(A, B):
            return Scatter([(B, 2)], label=A)
        dmap = DynamicMap(fn, kdims=['A-dim', 'B-dim'])
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_swapped_by_name(self):

        def fn(A, B):
            return Scatter([(B, 2)], label=A)
        dmap = DynamicMap(fn, kdims=['B', 'A'])
        self.assertEqual(dmap[1, 'Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_kdims_only_invalid(self):

        def fn(A, B):
            return Scatter([(B, 2)], label=A)
        regexp = "Callable 'fn' accepts more positional arguments than there are kdims and stream parameters"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(fn, kdims=['A'])

    def test_dynamic_kdims_args_only(self):

        def fn(*args):
            A, B = args
            return Scatter([(B, 2)], label=A)
        dmap = DynamicMap(fn, kdims=['A', 'B'])
        self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))

    def test_dynamic_streams_only_kwargs(self):

        def fn(x=1, y=2):
            return Scatter([(x, y)], label='default')
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        self.assertEqual(dmap[:], Scatter([(1, 2)], label='default'))

    def test_dynamic_streams_only_keywords(self):

        def fn(**kwargs):
            return Scatter([(kwargs['x'], kwargs['y'])], label='default')
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=[], streams=[xy])
        self.assertEqual(dmap[:], Scatter([(1, 2)], label='default'))

    def test_dynamic_split_kdims_and_streams(self):

        def fn(A, x=1, y=2):
            return Scatter([(x, y)], label=A)
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_split_kdims_and_streams_invalid(self):

        def fn(x=1, y=2, B='default'):
            return Scatter([(x, y)], label=B)
        xy = streams.PointerXY(x=1, y=2)
        regexp = "Callback 'fn' signature over (.+?) does not accommodate required kdims"
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(fn, kdims=['A'], streams=[xy])

    def test_dynamic_split_mismatched_kdims(self):

        def fn(B, x=1, y=2):
            return Scatter([(x, y)], label=B)
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_split_mismatched_kdims_invalid(self):

        def fn(x, y, B):
            return Scatter([(x, y)], label=B)
        xy = streams.PointerXY(x=1, y=2)
        regexp = 'Unmatched positional kdim arguments only allowed at the start of the signature'
        with self.assertRaisesRegex(KeyError, regexp):
            DynamicMap(fn, kdims=['A'], streams=[xy])

    def test_dynamic_split_args_and_kwargs(self):

        def fn(*args, **kwargs):
            return Scatter([(kwargs['x'], kwargs['y'])], label=args[0])
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_all_keywords(self):

        def fn(A='default', x=1, y=2):
            return Scatter([(x, y)], label=A)
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_keywords_and_kwargs(self):

        def fn(A='default', x=1, y=2, **kws):
            return Scatter([(x, y)], label=A)
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))

    def test_dynamic_mixed_kwargs(self):

        def fn(x, A, y):
            return Scatter([(x, y)], label=A)
        xy = streams.PointerXY(x=1, y=2)
        dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
        self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))