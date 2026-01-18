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
def QuoteOperand(self, operand, always=False):
    """Returns operand enclosed in "..." if necessary.

    Args:
      operand: A string operand or list of string operands. If a list then each
        list item is quoted.
      always: Always quote if True.

    Returns:
      A string: operand enclosed in "..." if necessary.
    """
    if isinstance(operand, list):
        operands = [self.Quote(x, always=always) for x in operand]
        return '(' + ','.join([x for x in operands if x is not None]) + ')'
    return self.Quote(operand, always=always)