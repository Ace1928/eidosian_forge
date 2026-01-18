from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IInterfaceDeclaration(Interface):
    """
    Declare and check the interfaces of objects.

    The functions defined in this interface are used to declare the
    interfaces that objects provide and to query the interfaces that
    have been declared.

    Interfaces can be declared for objects in two ways:

        - Interfaces are declared for instances of the object's class

        - Interfaces are declared for the object directly.

    The interfaces declared for an object are, therefore, the union of
    interfaces declared for the object directly and the interfaces
    declared for instances of the object's class.

    Note that we say that a class implements the interfaces provided
    by it's instances. An instance can also provide interfaces
    directly. The interfaces provided by an object are the union of
    the interfaces provided directly and the interfaces implemented by
    the class.

    This interface is implemented by :mod:`zope.interface`.
    """
    Interface = Attribute('The base class used to create new interfaces')

    def taggedValue(key, value):
        """
        Attach a tagged value to an interface while defining the interface.

        This is a way of executing :meth:`IElement.setTaggedValue` from
        the definition of the interface. For example::

             class IFoo(Interface):
                 taggedValue('key', 'value')

        .. seealso:: `zope.interface.taggedValue`
        """

    def invariant(checker_function):
        """
        Attach an invariant checker function to an interface while defining it.

        Invariants can later be validated against particular implementations by
        calling :meth:`IInterface.validateInvariants`.

        For example::

             def check_range(ob):
                 if ob.max < ob.min:
                     raise ValueError("max value is less than min value")

             class IRange(Interface):
                 min = Attribute("The min value")
                 max = Attribute("The max value")

                 invariant(check_range)

        .. seealso:: `zope.interface.invariant`
        """

    def interfacemethod(method):
        """
        A decorator that transforms a method specification into an
        implementation method.

        This is used to override methods of ``Interface`` or provide new methods.
        Definitions using this decorator will not appear in :meth:`IInterface.names()`.
        It is possible to have an implementation method and a method specification
        of the same name.

        For example::

             class IRange(Interface):
                 @interfacemethod
                 def __adapt__(self, obj):
                     if isinstance(obj, range):
                         # Return the builtin ``range`` as-is
                         return obj
                     return super(type(IRange), self).__adapt__(obj)

        You can use ``super`` to call the parent class functionality. Note that
        the zero-argument version (``super().__adapt__``) works on Python 3.6 and above, but
        prior to that the two-argument version must be used, and the class must be explicitly
        passed as the first argument.

        .. versionadded:: 5.1.0
        .. seealso:: `zope.interface.interfacemethod`
        """

    def providedBy(ob):
        """
        Return the interfaces provided by an object.

        This is the union of the interfaces directly provided by an
        object and interfaces implemented by it's class.

        The value returned is an `IDeclaration`.

        .. seealso:: `zope.interface.providedBy`
        """

    def implementedBy(class_):
        """
        Return the interfaces implemented for a class's instances.

        The value returned is an `IDeclaration`.

        .. seealso:: `zope.interface.implementedBy`
        """

    def classImplements(class_, *interfaces):
        """
        Declare additional interfaces implemented for instances of a class.

        The arguments after the class are one or more interfaces or
        interface specifications (`IDeclaration` objects).

        The interfaces given (including the interfaces in the
        specifications) are added to any interfaces previously
        declared.

        Consider the following example::

          class C(A, B):
             ...

          classImplements(C, I1, I2)


        Instances of ``C`` provide ``I1``, ``I2``, and whatever interfaces
        instances of ``A`` and ``B`` provide. This is equivalent to::

            @implementer(I1, I2)
            class C(A, B):
                pass

        .. seealso:: `zope.interface.classImplements`
        .. seealso:: `zope.interface.implementer`
        """

    def classImplementsFirst(cls, interface):
        """
        See :func:`zope.interface.classImplementsFirst`.
        """

    def implementer(*interfaces):
        """
        Create a decorator for declaring interfaces implemented by a
        factory.

        A callable is returned that makes an implements declaration on
        objects passed to it.

        .. seealso:: :meth:`classImplements`
        """

    def classImplementsOnly(class_, *interfaces):
        """
        Declare the only interfaces implemented by instances of a class.

        The arguments after the class are one or more interfaces or
        interface specifications (`IDeclaration` objects).

        The interfaces given (including the interfaces in the
        specifications) replace any previous declarations.

        Consider the following example::

          class C(A, B):
             ...

          classImplements(C, IA, IB. IC)
          classImplementsOnly(C. I1, I2)

        Instances of ``C`` provide only ``I1``, ``I2``, and regardless of
        whatever interfaces instances of ``A`` and ``B`` implement.

        .. seealso:: `zope.interface.classImplementsOnly`
        """

    def implementer_only(*interfaces):
        """
        Create a decorator for declaring the only interfaces implemented.

        A callable is returned that makes an implements declaration on
        objects passed to it.

        .. seealso:: `zope.interface.implementer_only`
        """

    def directlyProvidedBy(object):
        """
        Return the interfaces directly provided by the given object.

        The value returned is an `IDeclaration`.

        .. seealso:: `zope.interface.directlyProvidedBy`
        """

    def directlyProvides(object, *interfaces):
        """
        Declare interfaces declared directly for an object.

        The arguments after the object are one or more interfaces or
        interface specifications (`IDeclaration` objects).

        .. caution::
           The interfaces given (including the interfaces in the
           specifications) *replace* interfaces previously
           declared for the object. See :meth:`alsoProvides` to add
           additional interfaces.

        Consider the following example::

          class C(A, B):
             ...

          ob = C()
          directlyProvides(ob, I1, I2)

        The object, ``ob`` provides ``I1``, ``I2``, and whatever interfaces
        instances have been declared for instances of ``C``.

        To remove directly provided interfaces, use `directlyProvidedBy` and
        subtract the unwanted interfaces. For example::

          directlyProvides(ob, directlyProvidedBy(ob)-I2)

        removes I2 from the interfaces directly provided by
        ``ob``. The object, ``ob`` no longer directly provides ``I2``,
        although it might still provide ``I2`` if it's class
        implements ``I2``.

        To add directly provided interfaces, use `directlyProvidedBy` and
        include additional interfaces.  For example::

          directlyProvides(ob, directlyProvidedBy(ob), I2)

        adds I2 to the interfaces directly provided by ob.

        .. seealso:: `zope.interface.directlyProvides`
        """

    def alsoProvides(object, *interfaces):
        """
        Declare additional interfaces directly for an object.

        For example::

          alsoProvides(ob, I1)

        is equivalent to::

          directlyProvides(ob, directlyProvidedBy(ob), I1)

        .. seealso:: `zope.interface.alsoProvides`
        """

    def noLongerProvides(object, interface):
        """
        Remove an interface from the list of an object's directly provided
        interfaces.

        For example::

          noLongerProvides(ob, I1)

        is equivalent to::

          directlyProvides(ob, directlyProvidedBy(ob) - I1)

        with the exception that if ``I1`` is an interface that is
        provided by ``ob`` through the class's implementation,
        `ValueError` is raised.

        .. seealso:: `zope.interface.noLongerProvides`
        """

    def provider(*interfaces):
        """
        Declare interfaces provided directly by a class.

        .. seealso:: `zope.interface.provider`
        """

    def moduleProvides(*interfaces):
        """
        Declare interfaces provided by a module.

        This function is used in a module definition.

        The arguments are one or more interfaces or interface
        specifications (`IDeclaration` objects).

        The given interfaces (including the interfaces in the
        specifications) are used to create the module's direct-object
        interface specification.  An error will be raised if the module
        already has an interface specification.  In other words, it is
        an error to call this function more than once in a module
        definition.

        This function is provided for convenience. It provides a more
        convenient way to call `directlyProvides` for a module. For example::

          moduleImplements(I1)

        is equivalent to::

          directlyProvides(sys.modules[__name__], I1)

        .. seealso:: `zope.interface.moduleProvides`
        """

    def Declaration(*interfaces):
        """
        Create an interface specification.

        The arguments are one or more interfaces or interface
        specifications (`IDeclaration` objects).

        A new interface specification (`IDeclaration`) with the given
        interfaces is returned.

        .. seealso:: `zope.interface.Declaration`
        """