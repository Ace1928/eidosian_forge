import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
class InterfaceMeta(type):

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        classdict.setdefault('_next_id', 0)
        classdict.setdefault('_plugins', {})
        classdict.setdefault('_aliases', {})
        return super().__new__(cls, name, bases, classdict, *args, **kwargs)