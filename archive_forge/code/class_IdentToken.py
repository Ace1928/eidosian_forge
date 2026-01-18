from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class IdentToken(Node):
    """An :diagram:`ident-token`.

    .. autoattribute:: type

    .. attribute:: value

        The unescaped value, as a Unicode string.

    .. attribute:: lower_value

        Same as :attr:`value` but normalized to *ASCII lower case*,
        see :func:`~webencodings.ascii_lower`.
        This is the value to use when comparing to a CSS keyword.

    """
    __slots__ = ['value', 'lower_value']
    type = 'ident'
    repr_format = '<{self.__class__.__name__} {self.value}>'

    def __init__(self, line, column, value):
        Node.__init__(self, line, column)
        self.value = value
        try:
            self.lower_value = ascii_lower(value)
        except UnicodeEncodeError:
            self.lower_value = value

    def _serialize_to(self, write):
        write(serialize_identifier(self.value))