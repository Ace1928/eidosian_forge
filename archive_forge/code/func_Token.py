from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def Token(self, terminators='', balance_parens=False, space=True, convert=False):
    """Parses a possibly quoted token from the current expression position.

    The quote characters are in _QUOTES. The _ESCAPE character can prefix
    an _ESCAPE or _QUOTE character to treat it as a normal character. If
    _ESCAPE is at end of input, or is followed by any other character, then it
    is treated as a normal character.

    Quotes may be adjacent ("foo"" & ""bar" => "foo & bar") and they may appear
    mid token (foo" & "bar => "foo & bar").

    Args:
      terminators: A set of characters that terminate the token. isspace()
        characters always terminate the token. It may be a string, tuple, list
        or set. Terminator characters are not consumed.
      balance_parens: True if (...) must be balanced.
      space: True if space characters should be skipped after the token. Space
        characters are always skipped before the token.
      convert: Converts unquoted numeric string tokens to numbers if True.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.

    Returns:
      None if there is no token, the token string if convert is False or the
      token is quoted, otherwise the converted float / int / string value of
      the token.
    """
    quote = None
    quoted = False
    token = None
    paren_count = 0
    i = self.GetPosition()
    while not self.EndOfInput(i):
        c = self._expr[i]
        if c == self._ESCAPE and (not self.EndOfInput(i + 1)):
            c = self._expr[i + 1]
            if token is None:
                token = []
            if c != self._ESCAPE and c != quote and (quote or c not in self._QUOTES):
                token.append(self._ESCAPE)
            token.append(c)
            i += 1
        elif c == quote:
            quote = None
        elif not quote and c in self._QUOTES:
            quote = c
            quoted = True
            if token is None:
                token = []
        elif not quote and c.isspace() and (token is None):
            pass
        elif not quote and balance_parens and (c in '()'):
            if c == '(':
                paren_count += 1
            else:
                if c in terminators and (not paren_count):
                    break
                paren_count -= 1
            if token is None:
                token = []
            token.append(c)
        elif not quote and (not paren_count) and (c in terminators):
            break
        elif quote or not c.isspace() or (token is not None and balance_parens):
            if token is None:
                token = []
            token.append(c)
        elif token is not None:
            break
        i += 1
    if quote:
        raise resource_exceptions.ExpressionSyntaxError('Unterminated [{0}] quote [{1}].'.format(quote, self.Annotate()))
    self.SetPosition(i)
    if space:
        self.SkipSpace(terminators=terminators)
    if token is not None:
        token = ''.join(token)
    if convert and token and (not quoted):
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                pass
    return token