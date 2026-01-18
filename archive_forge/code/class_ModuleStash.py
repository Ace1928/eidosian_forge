import sys
import pytest
class ModuleStash(object):
    """
    Stashes away previously imported modules

    If we reimport a module the data from coverage is lost, so we reuse the old
    modules
    """

    def __init__(self, namespace, modules=sys.modules):
        self.namespace = namespace
        self.modules = modules
        self._data = {}

    def stash(self):
        self._data[self.namespace] = self.modules.pop(self.namespace, None)
        for module in list(self.modules.keys()):
            if module.startswith(self.namespace + '.'):
                self._data[module] = self.modules.pop(module)

    def pop(self):
        self.modules.pop(self.namespace, None)
        for module in list(self.modules.keys()):
            if module.startswith(self.namespace + '.'):
                self.modules.pop(module)
        self.modules.update(self._data)