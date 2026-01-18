from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
class TestKwargCallableInvocation(ComparisonTestCase):
    """
    Test invocation of Callable with kwargs, even for callbacks with
    positional arguments.
    """

    def test_callable_fn(self):

        def callback(x):
            return x
        self.assertEqual(Callable(callback)(x=3), 3)

    def test_callable_lambda(self):
        self.assertEqual(Callable(lambda x, y: x + y)(x=3, y=5), 8)

    def test_callable_partial(self):
        self.assertEqual(Callable(partial(lambda x, y: x + y, x=4))(y=5), 9)

    def test_callable_instance_method(self):
        assert Callable(CallableClass().someinstancemethod)(x=1, y=2) == 3

    def test_callable_partial_instance_method(self):
        assert Callable(partial(CallableClass().someinstancemethod, x=1))(y=2) == 3

    def test_callable_paramfunc(self):
        self.assertEqual(Callable(ParamFunc)(a=3, b=5), 15)