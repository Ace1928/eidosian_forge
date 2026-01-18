from ..utils import DummyObject, requires_backends
class SpeechEncoderDecoderModel(metaclass=DummyObject):
    _backends = ['torch']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['torch'])