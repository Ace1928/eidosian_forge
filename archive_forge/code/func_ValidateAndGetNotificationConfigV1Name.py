import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def ValidateAndGetNotificationConfigV1Name(args):
    """Returns relative resource name for a v1 notification config.

  Validates on regexes for args containing full names or short names with
  resources. Localization is supported by the
  ValidateAndGetNotificationConfigV2Name method.

  Args:
    args: an argparse object that should contain .NOTIFICATIONCONFIGID,
      optionally 1 of .organization, .folder, .project

  Examples:

  args with NOTIFICATIONCONFIGID="organizations/123/notificationConfigs/config1"
  returns the NOTIFICATIONCONFIGID

  args with NOTIFICATIONCONFIGID="config1" and projects="projects/123" returns
  projects/123/notificationConfigs/config1
  """
    resource_pattern = re.compile('(organizations|projects|folders)/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128}$')
    id_pattern = re.compile('[a-zA-Z0-9-_]{1,128}$')
    notification_config_id = args.NOTIFICATIONCONFIGID
    if not resource_pattern.match(notification_config_id) and (not id_pattern.match(notification_config_id)):
        raise errors.InvalidNotificationConfigError('NotificationConfig must match either (organizations|projects|folders)/.+/notificationConfigs/[a-zA-Z0-9-_]{1,128})$ or [a-zA-Z0-9-_]{1,128}$.')
    if resource_pattern.match(notification_config_id):
        return notification_config_id
    return util.GetParentFromNamedArguments(args) + '/notificationConfigs/' + notification_config_id