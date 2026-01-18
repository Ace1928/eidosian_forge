import inspect
from weakref import ref as weakref_ref
from pyomo.common.errors import PyomoException
from pyomo.common.deprecation import deprecated, deprecation_warning
class tmp_meta(cls, type(interface)):

    def __new__(cls, name, bases, classdict, *args, **kwargs):
        classdict.setdefault('_plugins', None)
        classdict.setdefault('_aliases', None)
        return super().__new__(cls, name, bases, classdict, *args, **kwargs)