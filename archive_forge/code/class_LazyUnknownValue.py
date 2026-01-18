from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
class LazyUnknownValue(AbstractLazyValue):

    def __init__(self, min=1, max=1):
        super().__init__(None, min, max)

    def infer(self):
        return NO_VALUES