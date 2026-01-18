from jedi.inference.base_value import ValueWrapper
from jedi.inference.value.module import ModuleValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import StubName, StubModuleName
from jedi.inference.gradual.typing import TypingModuleFilterWrapper
from jedi.inference.context import ModuleContext
class TypingModuleContext(ModuleContext):

    def get_filters(self, *args, **kwargs):
        filters = super().get_filters(*args, **kwargs)
        yield TypingModuleFilterWrapper(next(filters, None))
        yield from filters