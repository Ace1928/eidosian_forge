from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def CreateWaitText(igm_ref):
    """Creates text presented at each wait operation.

  Args:
    igm_ref: reference to the Instance Group Manager.
  Returns:
    A message with current operations count for IGM.
  """
    text = 'Waiting for group to become stable'
    current_actions_text = _CreateActionsText(', current operations: ', igm_ref.currentActions, _CURRENT_ACTION_TYPES)
    return text + current_actions_text