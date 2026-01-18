from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def GetProjectDeletionRisk(release_track, project_id):
    """Returns a risk assesment message for project deletion.

  Args:
    release_track: Release track of the recommender.
    project_id: Project ID.

  Returns:
    String message prompt to be displayed for project deletion.
    If the project deletion is high risk, the message includes the
    Active Assist warning.
  """
    risk_insight = _GetRiskInsight(release_track, project_id, _PROJECT_INSIGHT_TYPE)
    if risk_insight:
        return '\n'.join([_PROJECT_WARNING_MESSAGE, _GetDeletionRiskMessage(gcloud_insight=risk_insight, risk_message=_PROJECT_RISK_MESSAGE, reasons_prefix=_PROJECT_REASONS_PREFIX), _PROJECT_ADVICE, _GetInsightLink(risk_insight)])
    return _PROJECT_WARNING_MESSAGE