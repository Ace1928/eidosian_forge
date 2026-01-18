from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.recommender import insight
from googlecloudsdk.command_lib.projects import util as project_util
def GetIamPolicyBindingDeletionRisk(release_track, project_id, member, member_role):
    """Returns a risk assesment message for IAM policy binding deletion.

  Args:
    release_track: Release track of the recommender.
    project_id: String project ID.
    member: IAM policy binding member.
    member_role: IAM policy binding member role.

  Returns:
    String Active Assist risk warning message to be displayed in IAM policy
    binding deletion prompt.
    If no risk exists, then returns 'None'.
  """
    member = member[member.find(':') + 1:]
    policy_matcher = _GetIamPolicyBindingMatcher(member, member_role)
    risk_insight = _GetRiskInsight(release_track, project_id, _POLICY_BINDING_INSIGHT_TYPE, matcher=policy_matcher)
    if risk_insight:
        risk_message = '{} {}'.format(_GetDeletionRiskMessage(risk_insight, _POLICY_BINDING_DELETE_RISK_MESSAGE.format(member_role), add_new_line=False), _GetPolicyBindingDeletionAdvice(_GetPolicyBindingMinimalRoles(risk_insight)))
        return '\n'.join([_POLICY_BINDING_DELETE_WARNING_MESSAGE.format(member_role), risk_message, _GetInsightLink(risk_insight)])
    return _POLICY_BINDING_DELETE_WARNING_MESSAGE.format(member_role)