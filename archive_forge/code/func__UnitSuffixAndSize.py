from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def _UnitSuffixAndSize(unit):
    """Returns the unit size for unit, 1.0 for unknown units.

    Args:
      unit: The unit suffix (only the first character is checked), the unit
        size in bytes, or None.

    Returns:
      A (unit_suffix, unit_size) tuple.
    """
    unit_size = {'K': 2 ** 10, 'M': 2 ** 20, 'G': 2 ** 30, 'T': 2 ** 40, 'P': 2 ** 50}
    try:
        return ('', float(unit) or 1.0)
    except (TypeError, ValueError):
        pass
    try:
        unit_suffix = unit[0].upper()
        return (unit_suffix, unit_size[unit_suffix])
    except (IndexError, KeyError, TypeError):
        pass
    return ('', 1.0)