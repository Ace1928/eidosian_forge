from the server back to the client.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
def Parenthesize(self, expression):
    """Returns expression enclosed in (...) if it contains AND/OR."""
    lex = resource_lex.Lexer(expression)
    while True:
        tok = lex.Token(' ()', balance_parens=True)
        if not tok:
            break
        if tok in ['AND', 'OR']:
            return '({expression})'.format(expression=expression)
    return expression