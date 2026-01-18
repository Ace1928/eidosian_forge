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
def _ParseTerm(self, must=False):
    """Parses a [-]<key> <operator> <operand> term.

    Args:
      must: Raises ExpressionSyntaxError if must is True and there is no
        expression.

    Raises:
      ExpressionSyntaxError: The expression has a syntax error.

    Returns:
      The new backend expression tree.
    """
    here = self._lex.GetPosition()
    if not self._lex.SkipSpace():
        if must:
            raise resource_exceptions.ExpressionSyntaxError('Term expected [{0}].'.format(self._lex.Annotate(here)))
        return None
    if self._lex.IsCharacter(')', peek=True):
        return None
    if self._lex.IsCharacter('('):
        self._parenthesize.append(set())
        tree = self._ParseExpr()
        self._lex.IsCharacter(')')
        self._parenthesize.pop()
        return tree
    invert = self._lex.IsCharacter('-')
    here = self._lex.GetPosition()
    syntax_error = None
    try:
        key, transform = self._ParseKey()
        restriction = None
    except resource_exceptions.ExpressionSyntaxError as e:
        syntax_error = e
        self._lex.SetPosition(here)
        restriction = self._lex.Token(resource_lex.OPERATOR_CHARS, space=False)
        transform = None
        key = None
    here = self._lex.GetPosition()
    operator = self._ParseOperator()
    if not operator:
        if transform and (not key):
            tree = self._backend.ExprGlobal(transform)
        elif transform:
            raise resource_exceptions.ExpressionSyntaxError('Operator expected [{0}].'.format(self._lex.Annotate(here)))
        elif restriction in ['AND', 'OR']:
            raise resource_exceptions.ExpressionSyntaxError('Term expected [{0}].'.format(self._lex.Annotate()))
        elif isinstance(syntax_error, resource_exceptions.UnknownTransformError):
            raise syntax_error
        else:
            if not restriction:
                restriction = resource_lex.GetKeyName(key, quote=False)
            pattern = re.compile(re.escape(restriction), re.IGNORECASE)
            name = resource_projection_spec.GLOBAL_RESTRICTION_NAME
            tree = self._backend.ExprGlobal(resource_lex.MakeTransform(name, self._defaults.symbols.get(name, resource_property.EvaluateGlobalRestriction), args=[restriction, pattern]))
        if invert:
            tree = self._backend.ExprNOT(tree)
        return tree
    elif syntax_error:
        raise syntax_error
    self._lex.SkipSpace(token='Operand')
    here = self._lex.GetPosition()
    if any([self._lex.IsString(x) for x in self._LOGICAL]):
        raise resource_exceptions.ExpressionSyntaxError('Logical operator not expected [{0}].'.format(self._lex.Annotate(here)))
    if operator in (self._backend.ExprEQ, self._backend.ExprHAS) and self._lex.IsCharacter('('):
        operand = [arg for arg in self._lex.Args(separators=' \t\n,') if arg not in self._LOGICAL]
    else:
        operand = self._lex.Token('()')
    if operand is None:
        raise resource_exceptions.ExpressionSyntaxError('Term operand expected [{0}].'.format(self._lex.Annotate(here)))
    tree = operator(key=key, operand=self._backend.ExprOperand(operand), transform=transform)
    if invert:
        tree = self._backend.ExprNOT(tree)
    return tree