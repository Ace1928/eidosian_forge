from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import display_taps
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import cache_update_ops
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_reference
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import peek_iterable
import six
def _GetSortKeys(self):
    """Returns the list of --sort-by [(key, reverse)] tuples.

    Returns:
      The list of --sort-by [(key, reverse)] tuples, None if --sort-by was not
      specified. The keys are ordered from highest to lowest precedence.
    """
    if not self._GetFlag('sort_by'):
        return None
    keys = []
    for name in self._args.sort_by:
        if name.startswith('~'):
            name = name.lstrip('~')
            reverse = True
        else:
            reverse = False
        name = name.replace('[]', '[0]')
        keys.append((resource_lex.Lexer(name).Key(), reverse))
    return keys