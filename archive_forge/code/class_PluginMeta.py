import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
class PluginMeta(type):

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        _singleton = classdict.pop('__singleton__', any((getattr(base, '__singleton__', None) is not None for base in bases)))
        classdict['__singleton__'] = None
        aliases = classdict.setdefault('__plugin_aliases__', [])
        implements = classdict.setdefault('__implements__', [])
        interfaces = set((impl[0] for impl in implements))
        for base in bases:
            implements.extend((ep for ep in getattr(base, '__implements__', []) if ep[0] not in interfaces))
            interfaces.update((impl[0] for impl in implements))
        for interface, inherit, service in implements:
            if not inherit:
                continue
            if not any((issubclass(base, interface) for base in bases)):
                bases = bases + (interface,)
                if not issubclass(cls, type(interface)):

                    class tmp_meta(cls, type(interface)):

                        def __new__(cls, name, bases, classdict, *args, **kwargs):
                            classdict.setdefault('_plugins', None)
                            classdict.setdefault('_aliases', None)
                            return super().__new__(cls, name, bases, classdict, *args, **kwargs)
                    cls = tmp_meta
        new_class = super().__new__(cls, name, bases, classdict, *args, **kwargs)
        for interface, inherit, service in implements:
            interface._plugins[new_class] = {}
            interface._aliases.update({name: (new_class, doc) for name, doc in aliases})
        if _singleton:
            new_class.__singleton__ = new_class()
        return new_class