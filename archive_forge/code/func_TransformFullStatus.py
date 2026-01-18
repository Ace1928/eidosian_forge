from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def TransformFullStatus(r, undefined='UNKNOWN_ERROR'):
    """Returns a full description of the status of a logpoint or snapshot.

  Status will be one of ACTIVE, COMPLETED, or a verbose error description. If
  the status is an error, there will be additional information available in the
  status field of the object.

  Args:
    r: a JSON-serializable object
    undefined: Returns this value if the resource is not a valid status.

  Returns:
    One of ACTIVE, COMPLETED, or a verbose error description.

  Example:
    `--format="table(id, location, full_status())"`:::
    Displays the full status in the third table problem.
  """
    short_status, full_status = _TransformStatuses(r, undefined)
    if full_status:
        return '{0}: {1}'.format(short_status, full_status)
    else:
        return short_status