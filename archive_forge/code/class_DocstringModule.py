from jedi.inference.value import ModuleValue
from jedi.inference.context import ModuleContext
class DocstringModule(ModuleValue):

    def __init__(self, in_module_context, **kwargs):
        super().__init__(**kwargs)
        self._in_module_context = in_module_context

    def _as_context(self):
        return DocstringModuleContext(self, self._in_module_context)