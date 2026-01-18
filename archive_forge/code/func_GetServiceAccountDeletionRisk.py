from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def GetServiceAccountDeletionRisk(release_track, project_id, service_account):
    """Returns a risk assesment message for service account deletion.

  Args:
    release_track: Release track of the recommender.
    project_id: String project ID.
    service_account: Service Account email ID.

  Returns:
    String Active Assist risk warning message to be displayed in
    service account deletion prompt.
  """
    project_number = project_util.GetProjectNumber(project_id)
    if _IsDefaultAppEngineServiceAccount(service_account):
        return _SA_DELETE_APP_ENGINE_WARNING_MESSAGE
    if _IsDefaultComputeEngineServiceAccount(service_account, project_number):
        return _SA_DELETE_COMPUTE_ENGINE_WARNING_MESSAGE
    target_filter = 'targetResources: //iam.googleapis.com/projects/{0}/serviceAccounts/{1}'.format(project_number, service_account)
    risk_insight = _GetRiskInsight(release_track, project_id, _SA_INSIGHT_TYPE, request_filter=target_filter)
    if risk_insight:
        return '\n'.join([_SA_WARNING_MESSAGE, _GetDeletionRiskMessage(risk_insight, _SA_RISK_MESSAGE), _SA_ADVICE, _GetInsightLink(risk_insight)])
    return _SA_WARNING_MESSAGE