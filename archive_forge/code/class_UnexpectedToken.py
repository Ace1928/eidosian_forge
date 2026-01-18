from .utils import logger, NO_VALUE
from typing import Mapping, Iterable, Callable, Union, TypeVar, Tuple, Any, List, Set, Optional, Collection, TYPE_CHECKING
class UnexpectedToken(ParseError, UnexpectedInput):
    """An exception that is raised by the parser, when the token it received
    doesn't match any valid step forward.

    Parameters:
        token: The mismatched token
        expected: The set of expected tokens
        considered_rules: Which rules were considered, to deduce the expected tokens
        state: A value representing the parser state. Do not rely on its value or type.
        interactive_parser: An instance of ``InteractiveParser``, that is initialized to the point of failure,
                            and can be used for debugging and error handling.

    Note: These parameters are available as attributes of the instance.
    """
    expected: Set[str]
    considered_rules: Set[str]

    def __init__(self, token, expected, considered_rules=None, state=None, interactive_parser=None, terminals_by_name=None, token_history=None):
        super(UnexpectedToken, self).__init__()
        self.line = getattr(token, 'line', '?')
        self.column = getattr(token, 'column', '?')
        self.pos_in_stream = getattr(token, 'start_pos', None)
        self.state = state
        self.token = token
        self.expected = expected
        self._accepts = NO_VALUE
        self.considered_rules = considered_rules
        self.interactive_parser = interactive_parser
        self._terminals_by_name = terminals_by_name
        self.token_history = token_history

    @property
    def accepts(self) -> Set[str]:
        if self._accepts is NO_VALUE:
            self._accepts = self.interactive_parser and self.interactive_parser.accepts()
        return self._accepts

    def __str__(self):
        message = 'Unexpected token %r at line %s, column %s.\n%s' % (self.token, self.line, self.column, self._format_expected(self.accepts or self.expected))
        if self.token_history:
            message += 'Previous tokens: %r\n' % self.token_history
        return message