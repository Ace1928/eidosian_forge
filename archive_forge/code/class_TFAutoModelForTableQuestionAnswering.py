from ..utils import DummyObject, requires_backends
class TFAutoModelForTableQuestionAnswering(metaclass=DummyObject):
    _backends = ['tf']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['tf'])