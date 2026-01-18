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
def TransformSize(r, zero='0', precision=1, units_in=None, units_out=None, min=0):
    """Formats a human readable size in bytes.

  Args:
    r: A size in bytes.
    zero: Returns this if size==0. Ignored if None.
    precision: The number of digits displayed after the decimal point.
    units_in: A unit suffix (only the first character is checked) or unit size.
      The size is multiplied by this. The default is 1.0.
    units_out: A unit suffix (only the first character is checked) or unit size.
      The size is divided by this. The default is 1.0.
    min: Sizes < _min_ will be listed as "< _min_".

  Returns:
    A human readable scaled size in bytes.
  """

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
    if not r and zero is not None:
        return zero
    try:
        size = float(r)
    except (TypeError, ValueError):
        size = 0
    min_size = float(min)
    if size < min_size:
        size = min_size
        prefix = '< '
    else:
        prefix = ''
    _, units_in_size = _UnitSuffixAndSize(units_in)
    size *= units_in_size
    units_out_suffix, units_out_size = _UnitSuffixAndSize(units_out)
    if units_out_suffix:
        size /= units_out_size
        fmt = '{{0:.{precision}f}}'.format(precision=precision)
        return fmt.format(size)
    the_unit = 'PiB'
    for unit in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0:
            the_unit = unit
            break
        size /= 1024.0
    if the_unit:
        the_unit = ' ' + the_unit
    if size == int(size):
        return '{0}{1}{2}'.format(prefix, int(size), the_unit)
    else:
        fmt = '{{0}}{{1:.{precision}f}}{{2}}'.format(precision=precision)
        return fmt.format(prefix, size, the_unit)