from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class LiteralToken(Node):
    """Token that represents one or more characters as in the CSS source.

    .. autoattribute:: type

    .. attribute:: value

        A string of one to four characters.

    Instances compare equal to their :attr:`value`,
    so that these are equivalent:

    .. code-block:: python

        if node == ';':
        if node.type == 'literal' and node.value == ';':

    This regroups what `the specification`_ defines as separate token types:

    .. _the specification: https://drafts.csswg.org/css-syntax-3/

    * *<colon-token>* ``:``
    * *<semicolon-token>* ``;``
    * *<comma-token>* ``,``
    * *<cdc-token>* ``-->``
    * *<cdo-token>* ``<!--``
    * *<include-match-token>* ``~=``
    * *<dash-match-token>* ``|=``
    * *<prefix-match-token>* ``^=``
    * *<suffix-match-token>* ``$=``
    * *<substring-match-token>* ``*=``
    * *<column-token>* ``||``
    * *<delim-token>* (a single ASCII character not part of any another token)

    """
    __slots__ = ['value']
    type = 'literal'
    repr_format = '<{self.__class__.__name__} {self.value}>'

    def __init__(self, line, column, value):
        Node.__init__(self, line, column)
        self.value = value

    def __eq__(self, other):
        return self.value == other or self is other

    def __ne__(self, other):
        return not self == other

    def _serialize_to(self, write):
        write(self.value)