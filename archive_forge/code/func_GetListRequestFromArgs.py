from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetListRequestFromArgs(args, parent_resource, show_value):
    """Returns the get_request from the user-specified arguments.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
    parent_resource: resource location such as `organizations/123`
    show_value: if true, show all setting values set on the resource; if false,
      show all available settings.
  """
    messages = settings_service.ResourceSettingsMessages()
    if args.organization:
        view = messages.ResourcesettingsOrganizationsSettingsListRequest.ViewValueValuesEnum.SETTING_VIEW_LOCAL_VALUE if show_value else messages.ResourcesettingsOrganizationsSettingsListRequest.ViewValueValuesEnum.SETTING_VIEW_BASIC
        get_request = messages.ResourcesettingsOrganizationsSettingsListRequest(parent=parent_resource, view=view)
    elif args.folder:
        view = messages.ResourcesettingsFoldersSettingsListRequest.ViewValueValuesEnum.SETTING_VIEW_LOCAL_VALUE if show_value else messages.ResourcesettingsFoldersSettingsListRequest.ViewValueValuesEnum.SETTING_VIEW_BASIC
        get_request = messages.ResourcesettingsFoldersSettingsListRequest(parent=parent_resource, view=view)
    else:
        view = messages.ResourcesettingsProjectsSettingsListRequest.ViewValueValuesEnum.SETTING_VIEW_LOCAL_VALUE if show_value else messages.ResourcesettingsProjectsSettingsListRequest.ViewValueValuesEnum.SETTING_VIEW_BASIC
        get_request = messages.ResourcesettingsProjectsSettingsListRequest(parent=parent_resource, view=view)
    return get_request