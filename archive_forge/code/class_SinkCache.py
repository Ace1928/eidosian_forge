from ..utils import DummyObject, requires_backends
class SinkCache(metaclass=DummyObject):
    _backends = ['torch']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['torch'])