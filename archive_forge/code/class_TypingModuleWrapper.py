from jedi.inference.base_value import ValueWrapper
from jedi.inference.value.module import ModuleValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import StubName, StubModuleName
from jedi.inference.gradual.typing import TypingModuleFilterWrapper
from jedi.inference.context import ModuleContext
class TypingModuleWrapper(StubModuleValue):

    def get_filters(self, *args, **kwargs):
        filters = super().get_filters(*args, **kwargs)
        f = next(filters, None)
        assert f is not None
        yield TypingModuleFilterWrapper(f)
        yield from filters

    def _as_context(self):
        return TypingModuleContext(self)