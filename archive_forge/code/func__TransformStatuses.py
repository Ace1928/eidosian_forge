from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def _TransformStatuses(r, undefined):
    """Returns a full description of the status of a logpoint or snapshot.

  Status will be one of ACTIVE, COMPLETED, or a verbose error description. If
  the status is an error, there will be additional information available in the
  status field of the object.

  Args:
    r: a JSON-serializable object
    undefined: Returns this value if the resource is not a valid status.

  Returns:
    String, String - The first string will be a short error description,
    and the second a more detailed description.
  """
    short_status = undefined
    if isinstance(r, dict):
        if not r.get('isFinalState'):
            return ('ACTIVE', None)
        status = r.get('status')
        if not status or not isinstance(status, dict) or (not status.get('isError')):
            return ('COMPLETED', None)
        refers_to = status.get('refersTo')
        description = status.get('description')
        if refers_to:
            short_status = '{0}_ERROR'.format(refers_to).replace('BREAKPOINT_', '')
        if description:
            fmt = description.get('format')
            params = description.get('parameters') or []
            try:
                return (short_status, _SubstituteErrorParams(fmt, params))
            except (IndexError, KeyError):
                return (short_status, 'Malformed status message: {0}'.format(status))
    return (short_status, None)