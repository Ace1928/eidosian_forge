from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import deploy_command_util
from googlecloudsdk.api_lib.app import exceptions
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as s_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.third_party.appengine.admin.tools.conversion import convert_yaml
import six
def DisplayProposedConfigDeployments(project, configs):
    """Prints the details of the proposed config deployments.

  Args:
    project: The name of the current project.
    configs: [yaml_parsing.ConfigYamlInfo], The configurations being
      deployed.
  """
    log.status.Print('Configurations to update:\n')
    for c in configs:
        log.status.Print(DEPLOY_CONFIG_MESSAGE_TEMPLATE.format(project=project, type=CONFIG_TYPES[c.config], descriptor=c.file))
        if c.name == yaml_parsing.ConfigYamlInfo.QUEUE:
            try:
                api_maybe_enabled = enable_api.IsServiceEnabled(project, 'cloudtasks.googleapis.com')
            except s_exceptions.ListServicesPermissionDeniedException:
                api_maybe_enabled = True
            if api_maybe_enabled:
                log.warning(QUEUE_TASKS_WARNING)