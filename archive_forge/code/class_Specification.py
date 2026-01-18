import sys
import weakref
from types import FunctionType
from types import MethodType
from typing import Union
from zope.interface import ro
from zope.interface._compat import _use_c_impl
from zope.interface.exceptions import Invalid
from zope.interface.ro import ro as calculate_ro
from zope.interface.declarations import implementedBy
from zope.interface.declarations import providedBy
from zope.interface.exceptions import BrokenImplementation
from zope.interface.exceptions import InvalidInterface
from zope.interface.declarations import _empty
class Specification(SpecificationBase):
    """Specifications

    An interface specification is used to track interface declarations
    and component registrations.

    This class is a base class for both interfaces themselves and for
    interface specifications (declarations).

    Specifications are mutable.  If you reassign their bases, their
    relations with other specifications are adjusted accordingly.
    """
    __slots__ = ()
    _ROOT = None
    isOrExtends = SpecificationBase.isOrExtends
    providedBy = SpecificationBase.providedBy

    def __init__(self, bases=()):
        self._dependents = None
        self._bases = ()
        self._implied = {}
        self._v_attrs = None
        self.__iro__ = ()
        self.__sro__ = ()
        self.__bases__ = tuple(bases)

    @property
    def dependents(self):
        if self._dependents is None:
            self._dependents = weakref.WeakKeyDictionary()
        return self._dependents

    def subscribe(self, dependent):
        self._dependents[dependent] = self.dependents.get(dependent, 0) + 1

    def unsubscribe(self, dependent):
        try:
            n = self._dependents[dependent]
        except TypeError:
            raise KeyError(dependent)
        n -= 1
        if not n:
            del self.dependents[dependent]
        else:
            assert n > 0
            self.dependents[dependent] = n

    def __setBases(self, bases):
        for b in self.__bases__:
            b.unsubscribe(self)
        self._bases = bases
        for b in bases:
            b.subscribe(self)
        self.changed(self)
    __bases__ = property(lambda self: self._bases, __setBases)
    _do_calculate_ro = calculate_ro

    def _calculate_sro(self):
        """
        Calculate and return the resolution order for this object, using its ``__bases__``.

        Ensures that ``Interface`` is always the last (lowest priority) element.
        """
        sro = self._do_calculate_ro(base_mros={b: b.__sro__ for b in self.__bases__})
        root = self._ROOT
        if root is not None and sro and (sro[-1] is not root):
            sro = [x for x in sro if x is not root]
            sro.append(root)
        return sro

    def changed(self, originally_changed):
        """
        We, or something we depend on, have changed.

        By the time this is called, the things we depend on,
        such as our bases, should themselves be stable.
        """
        self._v_attrs = None
        implied = self._implied
        implied.clear()
        ancestors = self._calculate_sro()
        self.__sro__ = tuple(ancestors)
        self.__iro__ = tuple([ancestor for ancestor in ancestors if isinstance(ancestor, InterfaceClass)])
        for ancestor in ancestors:
            implied[ancestor] = ()
        for dependent in tuple(self._dependents.keys() if self._dependents else ()):
            dependent.changed(originally_changed)
        self._v_attrs = None

    def interfaces(self):
        """Return an iterator for the interfaces in the specification.
        """
        seen = {}
        for base in self.__bases__:
            for interface in base.interfaces():
                if interface not in seen:
                    seen[interface] = 1
                    yield interface

    def extends(self, interface, strict=True):
        """Does the specification extend the given interface?

        Test whether an interface in the specification extends the
        given interface
        """
        return interface in self._implied and (not strict or self != interface)

    def weakref(self, callback=None):
        return weakref.ref(self, callback)

    def get(self, name, default=None):
        """Query for an attribute description
        """
        attrs = self._v_attrs
        if attrs is None:
            attrs = self._v_attrs = {}
        attr = attrs.get(name)
        if attr is None:
            for iface in self.__iro__:
                attr = iface.direct(name)
                if attr is not None:
                    attrs[name] = attr
                    break
        return default if attr is None else attr