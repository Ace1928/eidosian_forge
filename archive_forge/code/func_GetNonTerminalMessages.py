from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import sys
def GetNonTerminalMessages(conditions, ignore_retry=False):
    """Get messages for non-terminal subconditions.

  Only show a message for some non-terminal subconditions:
  - if severity == warning
  - if message is provided
  Non-terminal subconditions that aren't warnings are effectively neutral,
  so messages for these aren't included unless provided.

  Args:
    conditions: Conditions
    ignore_retry: bool, if True, ignores the "Retry" condition

  Returns:
    list(str) messages of non-terminal subconditions
  """
    messages = []
    for c in conditions.NonTerminalSubconditions():
        if ignore_retry and c == 'Retry':
            continue
        if conditions[c]['severity'] == _SEVERITY_WARNING:
            messages.append('{}: {}'.format(c, conditions[c]['message'] or 'Unknown Warning.'))
        elif conditions[c]['message']:
            messages.append('{}: {}'.format(c, conditions[c]['message']))
    return messages