from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import frozendict
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util as projects_api_util
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding as encoder
from googlecloudsdk.core.util import retry
import six
def PromptToEnableDataAccessAuditLogs(service):
    """Prompts to enable Data Access audit logs for the given service.

  If the console cannot prompt, a warning is logged instead.

  Args:
    service: The service to enable Data Access audit logs for.
  """
    project = GetProject()
    project_ref = projects_util.ParseProject(project)
    warning_msg = 'If audit logs are not fully enabled for [{}], your function may fail to receive some events.'.format(service)
    try:
        policy = projects_api.GetIamPolicy(project_ref)
    except apitools_exceptions.HttpForbiddenError:
        log.warning('You do not have permission to retrieve the IAM policy and check whether Data Access audit logs are enabled for [{}]. {}'.format(service, warning_msg))
        return
    audit_config = _LookupAuditConfig(policy, service)
    enabled_log_types = set((lc.logType for lc in audit_config.auditLogConfigs))
    if enabled_log_types == _LOG_TYPES:
        return
    log.status.Print('Some Data Access audit logs are disabled for [{}]: https://console.cloud.google.com/iam-admin/audit?project={}'.format(service, project))
    if not console_io.CanPrompt():
        log.warning(warning_msg)
        return
    log.status.Print(warning_msg)
    if not console_io.PromptContinue(prompt_string='\nEnable all Data Access audit logs for [{}]?'.format(service)):
        return
    log_types_to_enable = [lt for lt in _LOG_TYPES if lt not in enabled_log_types]
    audit_config.auditLogConfigs.extend([_rm_messages.AuditLogConfig(logType=lt) for lt in log_types_to_enable])
    try:
        projects_api.SetIamPolicy(project_ref, policy, update_mask='auditConfigs')
        log.status.Print('Data Access audit logs successfully enabled.')
    except apitools_exceptions.HttpForbiddenError:
        log.warning('You do not have permission to update the IAM policy and ensure Data Access audit logs are enabled for [{}].'.format(service))