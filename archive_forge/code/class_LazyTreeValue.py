from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
class LazyTreeValue(AbstractLazyValue):

    def __init__(self, context, node, min=1, max=1):
        super().__init__(node, min, max)
        self.context = context
        self._predefined_names = dict(context.predefined_names)

    def infer(self):
        with monkeypatch(self.context, 'predefined_names', self._predefined_names):
            return self.context.infer_node(self.data)