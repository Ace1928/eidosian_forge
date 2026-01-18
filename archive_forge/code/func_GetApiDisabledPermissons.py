from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def GetApiDisabledPermissons(self):
    """Returns the API disabled permissions among the permissions provided."""
    api_disabled_permissions = []
    for permission in self.source_permissions:
        if permission in self.testable_permissions_map and self.testable_permissions_map[permission].customRolesSupportLevel != self.messages.Permission.CustomRolesSupportLevelValueValuesEnum.NOT_SUPPORTED and self.testable_permissions_map[permission].apiDisabled:
            api_disabled_permissions.append(permission)
    return api_disabled_permissions