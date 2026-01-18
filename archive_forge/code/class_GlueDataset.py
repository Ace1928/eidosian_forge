from ..utils import DummyObject, requires_backends
class GlueDataset(metaclass=DummyObject):
    _backends = ['torch']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['torch'])