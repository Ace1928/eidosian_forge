from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from dateutil import parser
from dateutil import tz
from dateutil.tz import _common as tz_common
import enum
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times_data
import six
def TzOffset(offset, name=None):
    """Returns a tzinfo for offset minutes east of UTC with optional name.

  Args:
    offset: The minutes east of UTC. Minutes west are negative.
    name: The optional timezone name. NOTE: no dst name.

  Returns:
    A tzinfo for offset seconds east of UTC.
  """
    return tz.tzoffset(name, offset * 60)