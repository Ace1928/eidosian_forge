import itertools
from typing import (
from zope.interface import implementer
from twisted.web.error import (
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
class Expose:
    """
    Helper for exposing methods for various uses using a simple decorator-style
    callable.

    Instances of this class can be called with one or more functions as
    positional arguments.  The names of these functions will be added to a list
    on the class object of which they are methods.
    """

    def __call__(self, f: _Tc, /, *funcObjs: Callable[..., object]) -> _Tc:
        """
        Add one or more functions to the set of exposed functions.

        This is a way to declare something about a class definition, similar to
        L{zope.interface.implementer}.  Use it like this::

            magic = Expose('perform extra magic')
            class Foo(Bar):
                def twiddle(self, x, y):
                    ...
                def frob(self, a, b):
                    ...
                magic(twiddle, frob)

        Later you can query the object::

            aFoo = Foo()
            magic.get(aFoo, 'twiddle')(x=1, y=2)

        The call to C{get} will fail if the name it is given has not been
        exposed using C{magic}.

        @param funcObjs: One or more function objects which will be exposed to
        the client.

        @return: The first of C{funcObjs}.
        """
        for fObj in itertools.chain([f], funcObjs):
            exposedThrough: List[Expose] = getattr(fObj, 'exposedThrough', [])
            exposedThrough.append(self)
            setattr(fObj, 'exposedThrough', exposedThrough)
        return f
    _nodefault = object()

    @overload
    def get(self, instance: object, methodName: str) -> Callable[..., Any]:
        ...

    @overload
    def get(self, instance: object, methodName: str, default: T) -> Union[Callable[..., Any], T]:
        ...

    def get(self, instance: object, methodName: str, default: object=_nodefault) -> object:
        """
        Retrieve an exposed method with the given name from the given instance.

        @raise UnexposedMethodError: Raised if C{default} is not specified and
        there is no exposed method with the given name.

        @return: A callable object for the named method assigned to the given
        instance.
        """
        method = getattr(instance, methodName, None)
        exposedThrough = getattr(method, 'exposedThrough', [])
        if self not in exposedThrough:
            if default is self._nodefault:
                raise UnexposedMethodError(self, methodName)
            return default
        return method