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
def Term(self, key, op, operand, transform, args):
    """Returns the rewritten backend term expression.

    Args:
      key: The parsed key.
      op: The operator name.
      operand: The operand.
      transform: The transform object if a transform was specified.
      args: The transform args if a transform was specified.

    Raises:
      UnknownFieldError: If key is not supported on the frontend and backend.

    Returns:
      The rewritten backend term expression.
    """
    if transform or args:
        return self.Expr(None)
    key_name = resource_lex.GetKeyName(key)
    if self.message:
        try:
            key_type, key = resource_property.GetMessageFieldType(key, self.message)
        except KeyError:
            if self.frontend_fields is not None and (not resource_property.LookupField(key, self.frontend_fields)):
                raise resource_exceptions.UnknownFieldError('Unknown field [{}] in expression.'.format(key_name))
            return self.Expr(None)
        else:
            key_name = resource_lex.GetKeyName(key)
    else:
        key_type = None
    return self.Expr(self.RewriteTerm(key_name, op, operand, key_type))