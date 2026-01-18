from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def GetMonitoringConfig(args):
    """Parses and validates the value of the managed prometheus config flags.

  Args:
    args: Arguments parsed from the command.

  Returns:
    The monitoring config object as GoogleCloudGkemulticloudV1MonitoringConfig.
    None if enable_managed_prometheus is None.
  """
    enabled_prometheus = getattr(args, 'enable_managed_prometheus', None)
    disabled_prometheus = getattr(args, 'disable_managed_prometheus', None)
    messages = api_util.GetMessagesModule()
    config = messages.GoogleCloudGkemulticloudV1ManagedPrometheusConfig()
    if enabled_prometheus:
        config.enabled = True
    elif disabled_prometheus:
        config.enabled = False
    else:
        return None
    return messages.GoogleCloudGkemulticloudV1MonitoringConfig(managedPrometheusConfig=config)