import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
class PluginFactory(object):

    def __init__(self, interface):
        self.interface = interface

    def __call__(self, name, *args, **kwds):
        name = str(name)
        if name not in self.interface._aliases:
            return None
        else:
            return self.interface._aliases[name][0](*args, **kwds)

    def services(self):
        return list(self.interface._aliases)

    def get_class(self, name):
        return self.interface._aliases.get(name, [None])[0]

    def doc(self, name):
        name = str(name)
        if name not in self.interface._aliases:
            return ''
        else:
            return self.interface._aliases[name][1]

    def deactivate(self, name):
        if isinstance(name, str):
            cls = self.get_class(name)
        if cls is None:
            return
        for service in ExtensionPoint(self.interface)(key=cls):
            service.deactivate()

    def activate(self, name):
        if isinstance(name, str):
            cls = self.get_class(name)
        if cls is None:
            return
        for service in ExtensionPoint(self.interface)(all=True, key=cls):
            service.activate()