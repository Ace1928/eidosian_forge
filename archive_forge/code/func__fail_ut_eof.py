import typing
import typing as t
from . import nodes
from .exceptions import TemplateAssertionError
from .exceptions import TemplateSyntaxError
from .lexer import describe_token
from .lexer import describe_token_expr
def _fail_ut_eof(self, name: t.Optional[str], end_token_stack: t.List[t.Tuple[str, ...]], lineno: t.Optional[int]) -> 'te.NoReturn':
    expected: t.Set[str] = set()
    for exprs in end_token_stack:
        expected.update(map(describe_token_expr, exprs))
    if end_token_stack:
        currently_looking: t.Optional[str] = ' or '.join(map(repr, map(describe_token_expr, end_token_stack[-1])))
    else:
        currently_looking = None
    if name is None:
        message = ['Unexpected end of template.']
    else:
        message = [f'Encountered unknown tag {name!r}.']
    if currently_looking:
        if name is not None and name in expected:
            message.append(f'You probably made a nesting mistake. Jinja is expecting this tag, but currently looking for {currently_looking}.')
        else:
            message.append(f'Jinja was looking for the following tags: {currently_looking}.')
    if self._tag_stack:
        message.append(f'The innermost block that needs to be closed is {self._tag_stack[-1]!r}.')
    self.fail(' '.join(message), lineno)