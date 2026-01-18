from ..utils import DummyObject, requires_backends
class GemmaTokenizer(metaclass=DummyObject):
    _backends = ['sentencepiece']

    def __init__(self, *args, **kwargs):
        requires_backends(self, ['sentencepiece'])