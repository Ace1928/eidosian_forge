from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def _GetDeletionRiskMessage(gcloud_insight, risk_message, reasons_prefix='', add_new_line=True):
    """Returns a risk message for resource deletion.

  Args:
    gcloud_insight: Insight object returned by the recommender API.
    risk_message: String risk message.
    reasons_prefix: String prefix before listing reasons.
    add_new_line: Bool for if a new line is added when no reasons are present.

  Returns:
    Formatted string risk message with reasons if any. The reasons are
    extracted from the gcloud_insight object.
  """
    reasons = _GetResourceRiskReasons(gcloud_insight)
    if not reasons:
        return '{}.{}'.format(risk_message, '\n' if add_new_line else '')
    if len(reasons) == 1:
        return '{0}{1} {2}\n'.format(risk_message, reasons_prefix, reasons[0])
    message = '{0}{1}:\n'.format(risk_message, reasons_prefix)
    message += ''.join(('  - {0}\n'.format(reason) for reason in reasons))
    return message