from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
class MergedLazyValues(AbstractLazyValue):
    """data is a list of lazy values."""

    def infer(self):
        return ValueSet.from_sets((l.infer() for l in self.data))