from __future__ import division, absolute_import
from functools import partial
from itertools import count
from operator import and_, or_, xor
class FlagConstant(_Constant):
    """
    L{FlagConstant} defines an attribute to be a flag constant within a
    collection defined by a L{Flags} subclass.

    L{FlagConstant} is only for use in the definition of L{Flags} subclasses.
    Do not instantiate L{FlagConstant} elsewhere and do not subclass it.
    """

    def __init__(self, value=_unspecified):
        _Constant.__init__(self)
        self.value = value

    def _realize(self, container, names, value):
        """
        Complete the initialization of this L{FlagConstant}.

        This implementation differs from other C{_realize} implementations in
        that a L{FlagConstant} may have several names which apply to it, due to
        flags being combined with various operators.

        @param container: The L{Flags} subclass this constant is part of.

        @param names: When a single-flag value is being initialized, a C{str}
            giving the name of that flag.  This is the case which happens when
            a L{Flags} subclass is being initialized and L{FlagConstant}
            instances from its body are being realized.  Otherwise, a C{set} of
            C{str} giving names of all the flags set on this L{FlagConstant}
            instance.  This is the case when two flags are combined using C{|},
            for example.
        """
        if isinstance(names, str):
            name = names
            names = set([names])
        elif len(names) == 1:
            name, = names
        else:
            name = '{' + ','.join(sorted(names)) + '}'
        _Constant._realize(self, container, name, value)
        self.value = value
        self.names = names

    def __or__(self, other):
        """
        Define C{|} on two L{FlagConstant} instances to create a new
        L{FlagConstant} instance with all flags set in either instance set.
        """
        return _flagOp(or_, self, other)

    def __and__(self, other):
        """
        Define C{&} on two L{FlagConstant} instances to create a new
        L{FlagConstant} instance with only flags set in both instances set.
        """
        return _flagOp(and_, self, other)

    def __xor__(self, other):
        """
        Define C{^} on two L{FlagConstant} instances to create a new
        L{FlagConstant} instance with only flags set on exactly one instance
        set.
        """
        return _flagOp(xor, self, other)

    def __invert__(self):
        """
        Define C{~} on a L{FlagConstant} instance to create a new
        L{FlagConstant} instance with all flags not set on this instance set.
        """
        result = FlagConstant()
        result._realize(self._container, set(), 0)
        for flag in self._container.iterconstants():
            if flag.value & self.value == 0:
                result |= flag
        return result

    def __iter__(self):
        """
        @return: An iterator of flags set on this instance set.
        """
        return (self._container.lookupByName(name) for name in self.names)

    def __contains__(self, flag):
        """
        @param flag: The flag to test for membership in this instance
            set.

        @return: C{True} if C{flag} is in this instance set, else
            C{False}.
        """
        return bool(flag & self)

    def __nonzero__(self):
        """
        @return: C{False} if this flag's value is 0, else C{True}.
        """
        return bool(self.value)
    __bool__ = __nonzero__