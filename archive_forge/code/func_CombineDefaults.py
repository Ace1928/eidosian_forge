from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import sys
def CombineDefaults(defaults):
    """Combines a list of defaults into a new defaults object.

  Args:
    defaults: An ordered list of ProjectionSpec objects to combine. alias and
      symbol names from higher index objects in the list take precedence.

  Returns:
    A new ProjectionSpec object that is a combination of the objects in the
    defaults list.
  """
    aliases = {}
    symbols = {}
    for default in defaults:
        if not default:
            continue
        if default.symbols:
            symbols.update(default.symbols)
        if default.aliases:
            aliases.update(default.aliases)
    return ProjectionSpec(symbols=symbols, aliases=aliases)