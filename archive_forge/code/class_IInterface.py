from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IInterface(ISpecification, IElement):
    """Interface objects

    Interface objects describe the behavior of an object by containing
    useful information about the object.  This information includes:

    - Prose documentation about the object.  In Python terms, this
      is called the "doc string" of the interface.  In this element,
      you describe how the object works in prose language and any
      other useful information about the object.

    - Descriptions of attributes.  Attribute descriptions include
      the name of the attribute and prose documentation describing
      the attributes usage.

    - Descriptions of methods.  Method descriptions can include:

        - Prose "doc string" documentation about the method and its
          usage.

        - A description of the methods arguments; how many arguments
          are expected, optional arguments and their default values,
          the position or arguments in the signature, whether the
          method accepts arbitrary arguments and whether the method
          accepts arbitrary keyword arguments.

    - Optional tagged data.  Interface objects (and their attributes and
      methods) can have optional, application specific tagged data
      associated with them.  Examples uses for this are examples,
      security assertions, pre/post conditions, and other possible
      information you may want to associate with an Interface or its
      attributes.

    Not all of this information is mandatory.  For example, you may
    only want the methods of your interface to have prose
    documentation and not describe the arguments of the method in
    exact detail.  Interface objects are flexible and let you give or
    take any of these components.

    Interfaces are created with the Python class statement using
    either `zope.interface.Interface` or another interface, as in::

      from zope.interface import Interface

      class IMyInterface(Interface):
        '''Interface documentation'''

        def meth(arg1, arg2):
            '''Documentation for meth'''

        # Note that there is no self argument

     class IMySubInterface(IMyInterface):
        '''Interface documentation'''

        def meth2():
            '''Documentation for meth2'''

    You use interfaces in two ways:

    - You assert that your object implement the interfaces.

      There are several ways that you can declare that an object
      provides an interface:

      1. Call `zope.interface.implementer` on your class definition.

      2. Call `zope.interface.directlyProvides` on your object.

      3. Call `zope.interface.classImplements` to declare that instances
         of a class implement an interface.

         For example::

           from zope.interface import classImplements

           classImplements(some_class, some_interface)

         This approach is useful when it is not an option to modify
         the class source.  Note that this doesn't affect what the
         class itself implements, but only what its instances
         implement.

    - You query interface meta-data. See the IInterface methods and
      attributes for details.

    """

    def names(all=False):
        """Get the interface attribute names

        Return a collection of the names of the attributes, including
        methods, included in the interface definition.

        Normally, only directly defined attributes are included. If
        a true positional or keyword argument is given, then
        attributes defined by base classes will be included.
        """

    def namesAndDescriptions(all=False):
        """Get the interface attribute names and descriptions

        Return a collection of the names and descriptions of the
        attributes, including methods, as name-value pairs, included
        in the interface definition.

        Normally, only directly defined attributes are included. If
        a true positional or keyword argument is given, then
        attributes defined by base classes will be included.
        """

    def __getitem__(name):
        """Get the description for a name

        If the named attribute is not defined, a `KeyError` is raised.
        """

    def direct(name):
        """Get the description for the name if it was defined by the interface

        If the interface doesn't define the name, returns None.
        """

    def validateInvariants(obj, errors=None):
        """Validate invariants

        Validate object to defined invariants.  If errors is None,
        raises first Invalid error; if errors is a list, appends all errors
        to list, then raises Invalid with the errors as the first element
        of the "args" tuple."""

    def __contains__(name):
        """Test whether the name is defined by the interface"""

    def __iter__():
        """Return an iterator over the names defined by the interface

        The names iterated include all of the names defined by the
        interface directly and indirectly by base interfaces.
        """
    __module__ = Attribute('The name of the module defining the interface')