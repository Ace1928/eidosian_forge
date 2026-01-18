from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.resource import resource_property
def _GetFlagAndDest(self, dest):
    """Returns the argument parser (flag_name, flag_dest) for dest.

    Args:
      dest: The resource argument dest name.

    Returns:
      Returns the argument parser (flag_name, flag_dest) for dest.
    """
    dests = []
    if self._prefix:
        dests.append(self.GetDest(dest, prefix=self._prefix))
    dests.append(dest)
    for flag_dest in dests:
        try:
            return (self._parsed_args.GetFlag(flag_dest), flag_dest)
        except parser_errors.UnknownDestinationException:
            pass
    return (None, None)