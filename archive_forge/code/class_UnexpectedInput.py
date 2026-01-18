from .utils import logger, NO_VALUE
from typing import Mapping, Iterable, Callable, Union, TypeVar, Tuple, Any, List, Set, Optional, Collection, TYPE_CHECKING
class UnexpectedInput(LarkError):
    """UnexpectedInput Error.

    Used as a base class for the following exceptions:

    - ``UnexpectedCharacters``: The lexer encountered an unexpected string
    - ``UnexpectedToken``: The parser received an unexpected token
    - ``UnexpectedEOF``: The parser expected a token, but the input ended

    After catching one of these exceptions, you may call the following helper methods to create a nicer error message.
    """
    line: int
    column: int
    pos_in_stream = None
    state: Any
    _terminals_by_name = None
    interactive_parser: 'InteractiveParser'

    def get_context(self, text: str, span: int=40) -> str:
        """Returns a pretty string pinpointing the error in the text,
        with span amount of context characters around it.

        Note:
            The parser doesn't hold a copy of the text it has to parse,
            so you have to provide it again
        """
        assert self.pos_in_stream is not None, self
        pos = self.pos_in_stream
        start = max(pos - span, 0)
        end = pos + span
        if not isinstance(text, bytes):
            before = text[start:pos].rsplit('\n', 1)[-1]
            after = text[pos:end].split('\n', 1)[0]
            return before + after + '\n' + ' ' * len(before.expandtabs()) + '^\n'
        else:
            before = text[start:pos].rsplit(b'\n', 1)[-1]
            after = text[pos:end].split(b'\n', 1)[0]
            return (before + after + b'\n' + b' ' * len(before.expandtabs()) + b'^\n').decode('ascii', 'backslashreplace')

    def match_examples(self, parse_fn: 'Callable[[str], Tree]', examples: Union[Mapping[T, Iterable[str]], Iterable[Tuple[T, Iterable[str]]]], token_type_match_fallback: bool=False, use_accepts: bool=True) -> Optional[T]:
        """Allows you to detect what's wrong in the input text by matching
        against example errors.

        Given a parser instance and a dictionary mapping some label with
        some malformed syntax examples, it'll return the label for the
        example that bests matches the current error. The function will
        iterate the dictionary until it finds a matching error, and
        return the corresponding value.

        For an example usage, see `examples/error_reporting_lalr.py`

        Parameters:
            parse_fn: parse function (usually ``lark_instance.parse``)
            examples: dictionary of ``{'example_string': value}``.
            use_accepts: Recommended to keep this as ``use_accepts=True``.
        """
        assert self.state is not None, 'Not supported for this exception'
        if isinstance(examples, Mapping):
            examples = examples.items()
        candidate = (None, False)
        for i, (label, example) in enumerate(examples):
            assert not isinstance(example, str), 'Expecting a list'
            for j, malformed in enumerate(example):
                try:
                    parse_fn(malformed)
                except UnexpectedInput as ut:
                    if ut.state == self.state:
                        if use_accepts and isinstance(self, UnexpectedToken) and isinstance(ut, UnexpectedToken) and (ut.accepts != self.accepts):
                            logger.debug('Different accepts with same state[%d]: %s != %s at example [%s][%s]' % (self.state, self.accepts, ut.accepts, i, j))
                            continue
                        if isinstance(self, (UnexpectedToken, UnexpectedEOF)) and isinstance(ut, (UnexpectedToken, UnexpectedEOF)):
                            if ut.token == self.token:
                                logger.debug('Exact Match at example [%s][%s]' % (i, j))
                                return label
                            if token_type_match_fallback:
                                if ut.token.type == self.token.type and (not candidate[-1]):
                                    logger.debug('Token Type Fallback at example [%s][%s]' % (i, j))
                                    candidate = (label, True)
                        if candidate[0] is None:
                            logger.debug('Same State match at example [%s][%s]' % (i, j))
                            candidate = (label, False)
        return candidate[0]

    def _format_expected(self, expected):
        if self._terminals_by_name:
            d = self._terminals_by_name
            expected = [d[t_name].user_repr() if t_name in d else t_name for t_name in expected]
        return 'Expected one of: \n\t* %s\n' % '\n\t* '.join(expected)