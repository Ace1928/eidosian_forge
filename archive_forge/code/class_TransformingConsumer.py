import logging
import threading
from grpc.framework.foundation import stream
class TransformingConsumer(stream.Consumer):
    """A stream.Consumer that passes a transformation of its input to another."""

    def __init__(self, transformation, downstream):
        self._transformation = transformation
        self._downstream = downstream

    def consume(self, value):
        self._downstream.consume(self._transformation(value))

    def terminate(self):
        self._downstream.terminate()

    def consume_and_terminate(self, value):
        self._downstream.consume_and_terminate(self._transformation(value))