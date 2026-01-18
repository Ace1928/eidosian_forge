from jedi.inference.base_value import ValueWrapper
from jedi.inference.value.module import ModuleValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import StubName, StubModuleName
from jedi.inference.gradual.typing import TypingModuleFilterWrapper
from jedi.inference.context import ModuleContext
class StubModuleContext(ModuleContext):

    def get_filters(self, until_position=None, origin_scope=None):
        return super().get_filters(origin_scope=origin_scope)