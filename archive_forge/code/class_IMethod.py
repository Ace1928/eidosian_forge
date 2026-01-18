from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IMethod(IAttribute):
    """Method attributes"""

    def getSignatureInfo():
        """Returns the signature information.

        This method returns a dictionary with the following string keys:

        - positional
            A sequence of the names of positional arguments.
        - required
            A sequence of the names of required arguments.
        - optional
            A dictionary mapping argument names to their default values.
        - varargs
            The name of the varargs argument (or None).
        - kwargs
            The name of the kwargs argument (or None).
        """

    def getSignatureString():
        """Return a signature string suitable for inclusion in documentation.

        This method returns the function signature string. For example, if you
        have ``def func(a, b, c=1, d='f')``, then the signature string is ``"(a, b,
        c=1, d='f')"``.
        """