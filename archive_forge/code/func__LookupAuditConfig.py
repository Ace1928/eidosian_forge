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
def _LookupAuditConfig(iam_policy, service):
    """Looks up the audit config for the given service.

  If no audit config is found, a new one is created and attached to the given
  policy.

  Args:
    iam_policy: The IAM policy to look through.
    service: The service to find the audit config for.

  Returns:
    The audit config for the given service or a blank new one if not found.
  """
    for ac in iam_policy.auditConfigs:
        if ac.service == service:
            return ac
    audit_config = _rm_messages.AuditConfig(service=service, auditLogConfigs=[])
    iam_policy.auditConfigs.append(audit_config)
    return audit_config