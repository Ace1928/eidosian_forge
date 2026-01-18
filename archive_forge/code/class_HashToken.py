from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class HashToken(Node):
    """A :diagram:`hash-token`.

    .. code-block:: text

        '#' <value>

    .. autoattribute:: type

    .. attribute:: value

        The unescaped value, as a Unicode string, without the preceding ``#``.

    .. attribute:: is_identifier

        A boolean, true if the CSS source for this token
        was ``#`` followed by a valid identifier.
        (Only such hash tokens are valid ID selectors.)

    """
    __slots__ = ['value', 'is_identifier']
    type = 'hash'
    repr_format = '<{self.__class__.__name__} #{self.value}>'

    def __init__(self, line, column, value, is_identifier):
        Node.__init__(self, line, column)
        self.value = value
        self.is_identifier = is_identifier

    def _serialize_to(self, write):
        write('#')
        if self.is_identifier:
            write(serialize_identifier(self.value))
        else:
            write(serialize_name(self.value))