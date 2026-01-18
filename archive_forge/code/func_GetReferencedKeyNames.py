from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_keys_expr
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
def GetReferencedKeyNames(filter_string=None, format_string=None, printer=None, defaults=None):
    """Returns the set of key names referenced by filter / format expressions.

  NOTICE: OnePlatform is forgiving on filter and format key reference name
  spelling.  Use resource_property.GetMatchingIndex() when verifying against
  resource dictionaries to handle camel and snake case spellings.

  Args:
    filter_string: The resource filter expression string.
    format_string: The resource format expression string.
    printer: The parsed format_string.
    defaults: The resource format and filter default projection.

  Raises:
    ValueError: If both format_string and printer are specified.

  Returns:
    The set of key names referenced by filter / format expressions.
  """
    keys = set()
    if printer:
        if format_string:
            raise ValueError('Cannot specify both format_string and printer.')
    elif format_string:
        printer = resource_printer.Printer(format_string, defaults=defaults)
        defaults = printer.column_attributes
    if printer:
        for col in printer.column_attributes.Columns():
            keys.add(resource_lex.GetKeyName(col.key, omit_indices=True))
    if filter_string:
        expr = resource_filter.Compile(filter_string, defaults=defaults, backend=resource_keys_expr.Backend())
        for key in expr.Evaluate(None):
            keys.add(resource_lex.GetKeyName(key, omit_indices=True))
    return keys