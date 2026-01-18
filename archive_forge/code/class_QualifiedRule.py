from webencodings import ascii_lower
from .serializer import _serialize_to, serialize_identifier, serialize_name
class QualifiedRule(Node):
    """A :diagram:`qualified rule`.

    .. code-block:: text

        <prelude> '{' <content> '}'

    The interpretation of qualified rules depend on their context.
    At the top-level of a stylesheet
    or in a conditional rule such as ``@media``,
    they are **style rules** where the :attr:`prelude` is Selectors list
    and the :attr:`content` is a list of property declarations.

    .. autoattribute:: type

    .. attribute:: prelude

        The rule’s prelude, the part before the {} block,
        as a list of :term:`component values`.

    .. attribute:: content

        The rule’s content, the part inside the {} block,
        as a list of :term:`component values`.

    """
    __slots__ = ['prelude', 'content']
    type = 'qualified-rule'
    repr_format = '<{self.__class__.__name__} … {{ … }}>'

    def __init__(self, line, column, prelude, content):
        Node.__init__(self, line, column)
        self.prelude = prelude
        self.content = content

    def _serialize_to(self, write):
        _serialize_to(self.prelude, write)
        write('{')
        _serialize_to(self.content, write)
        write('}')