import threading
from pytest import fixture, raises
import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context
class TestMethodDecorators(BaseZMQTestCase):

    @context()
    @socket(zmq.PUB)
    @socket(zmq.SUB)
    def multi_skts_method(self, ctx, pub, sub, foo='bar'):
        assert isinstance(self, TestMethodDecorators), self
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert foo == 'bar'
        assert pub.context is ctx
        assert sub.context is ctx
        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB

    def test_multi_skts_method(self):
        self.multi_skts_method()

    def test_multi_skts_method_other_args(self):

        @socket(zmq.PUB)
        @socket(zmq.SUB)
        def f(foo, pub, sub, bar=None):
            assert isinstance(pub, zmq.Socket), pub
            assert isinstance(sub, zmq.Socket), sub
            assert foo == 'mock'
            assert bar == 'fake'
            assert pub.context is zmq.Context.instance()
            assert sub.context is zmq.Context.instance()
            assert pub.type == zmq.PUB
            assert sub.type == zmq.SUB
        f('mock', bar='fake')