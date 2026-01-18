from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
class AbstractLazyValue:

    def __init__(self, data, min=1, max=1):
        self.data = data
        self.min = min
        self.max = max

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.data)

    def infer(self):
        raise NotImplementedError