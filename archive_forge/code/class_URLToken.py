from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class URLToken(Node):
    """An :diagram:`url-token`.

    .. code-block:: text

        'url(' <value> ')'

    .. autoattribute:: type

    .. attribute:: value

        The unescaped URL, as a Unicode string, without the ``url(`` and ``)``
        markers.

    """
    __slots__ = ['value', 'representation']
    type = 'url'
    repr_format = '<{self.__class__.__name__} {self.representation}>'

    def __init__(self, line, column, value, representation):
        Node.__init__(self, line, column)
        self.value = value
        self.representation = representation

    def _serialize_to(self, write):
        write(self.representation)