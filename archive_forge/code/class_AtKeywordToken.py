from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class AtKeywordToken(Node):
    """An :diagram:`at-keyword-token`.

    .. code-block:: text

        '@' <value>

    .. autoattribute:: type

    .. attribute:: value

        The unescaped value, as a Unicode string, without the preceding ``@``.

    .. attribute:: lower_value

        Same as :attr:`value` but normalized to *ASCII lower case*,
        see :func:`~webencodings.ascii_lower`.
        This is the value to use when comparing to a CSS at-keyword.

        .. code-block:: python

            if node.type == 'at-keyword' and node.lower_value == 'import':

    """
    __slots__ = ['value', 'lower_value']
    type = 'at-keyword'
    repr_format = '<{self.__class__.__name__} @{self.value}>'

    def __init__(self, line, column, value):
        Node.__init__(self, line, column)
        self.value = value
        try:
            self.lower_value = ascii_lower(value)
        except UnicodeEncodeError:
            self.lower_value = value

    def _serialize_to(self, write):
        write('@')
        write(serialize_identifier(self.value))