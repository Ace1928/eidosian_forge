from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
class LazyKnownValues(AbstractLazyValue):
    """data is a ValueSet."""

    def infer(self):
        return self.data