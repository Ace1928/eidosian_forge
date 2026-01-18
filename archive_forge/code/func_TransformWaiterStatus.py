from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def TransformWaiterStatus(r, undefined=''):
    """Returns a short description of the status of a waiter or waiter operation.

  Status will be one of WAITING, SUCCESS, FAILURE, or TIMEOUT.

  Args:
    r: a JSON-serializable object
    undefined: Returns this value if the resource status cannot be determined.

  Returns:
    One of WAITING, SUCCESS, FAILURE, or TIMEOUT

  Example:
    `--format="table(name, status())"`:::
    Displays the status in table column two.
  """
    if not isinstance(r, dict):
        return undefined
    if not r.get('done'):
        return 'WAITING'
    error = r.get('error')
    if not error:
        return 'SUCCESS'
    if error.get('code') == DEADLINE_EXCEEDED:
        return 'TIMEOUT'
    else:
        return 'FAILURE'