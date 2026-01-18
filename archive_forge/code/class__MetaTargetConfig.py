import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
class _MetaTargetConfig(type):
    """Metaclass for ``TargetConfig``.

    When a subclass of ``TargetConfig`` is created, all ``Option`` defined
    as class members will be parsed and corresponding getters, setters, and
    delters will be inserted.
    """

    def __init__(cls, name, bases, dct):
        """Invoked when subclass is created.

        Insert properties for each ``Option`` that are class members.
        All the options will be grouped inside the ``.options`` class
        attribute.
        """
        opts = {}
        for base_cls in reversed(bases):
            opts.update(base_cls.options)
        opts.update(cls.find_options(dct))
        cls.options = MappingProxyType(opts)

        def make_prop(name, option):

            def getter(self):
                return self._values.get(name, option.default)

            def setter(self, val):
                self._values[name] = option.type(val)

            def delter(self):
                del self._values[name]
            return property(getter, setter, delter, option.doc)
        for name, option in cls.options.items():
            setattr(cls, name, make_prop(name, option))

    def find_options(cls, dct):
        """Returns a new dict with all the items that are a mapping to an
        ``Option``.
        """
        return {k: v for k, v in dct.items() if isinstance(v, Option)}