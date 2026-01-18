from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _UpdateTypesForOutput(self, val):
    """Dig through a dict of list of primitives to help yaml output.

    Args:
      val: A dict, list, or primitive object.

    Returns:
      An updated version of val.
    """
    from googlecloudsdk.core.yaml import dict_like
    from googlecloudsdk.core.yaml import list_like
    if isinstance(val, six.string_types) and '\n' in val:
        return YamlPrinter._LiteralLines(val)
    if list_like(val):
        for i in range(len(val)):
            val[i] = self._UpdateTypesForOutput(val[i])
        return val
    if dict_like(val):
        for key in val:
            val[key] = self._UpdateTypesForOutput(val[key])
        return val
    return val