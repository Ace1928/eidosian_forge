import sys
from types import ModuleType
from twisted.trial.unittest import TestCase
class SetAsideModule:
    """
    L{SetAsideModule} is a context manager for temporarily removing a module
    from C{sys.modules}.

    @ivar name: The name of the module to remove.
    """

    def __init__(self, name):
        self.name = name

    def _unimport(self, name):
        """
        Find the given module and all of its hierarchically inferior modules in
        C{sys.modules}, remove them from it, and return whatever was found.
        """
        modules = {moduleName: module for moduleName, module in list(sys.modules.items()) if moduleName == self.name or moduleName.startswith(self.name + '.')}
        for name in modules:
            del sys.modules[name]
        return modules

    def __enter__(self):
        self.modules = self._unimport(self.name)

    def __exit__(self, excType, excValue, traceback):
        self._unimport(self.name)
        sys.modules.update(self.modules)