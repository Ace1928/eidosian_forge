from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_expr
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
import six
def _ParseOperator(self):
    """Parses an operator token.

    All operators match the RE [_operator_char_1][_operator_char_2]. Invalid
    operators are 2 character sequences that are not valid operators and
    match the RE [_operator_char_1][_operator_char_1+_operator_char_2].

    Raises:
      ExpressionSyntaxError: The operator spelling is malformed.

    Returns:
      The operator backend expression, None if the next token is not an
      operator.
    """
    if not self._lex.SkipSpace():
        return None
    here = self._lex.GetPosition()
    op = self._lex.IsCharacter(self._operator_char_1)
    if not op:
        return None
    if not self._lex.EndOfInput():
        o2 = self._lex.IsCharacter(self._operator_char_1 + self._operator_char_2)
        if o2:
            op += o2
    if op not in self._operator:
        raise resource_exceptions.ExpressionSyntaxError('Malformed operator [{0}].'.format(self._lex.Annotate(here)))
    self._lex.SkipSpace(token='Term operand')
    return self._operator[op]