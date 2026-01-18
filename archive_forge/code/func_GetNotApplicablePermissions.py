from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def GetNotApplicablePermissions(self):
    """Returns the not applicable permissions among the permissions provided."""
    not_applicable_permissions = []
    for permission in self.source_permissions:
        if permission not in self.testable_permissions_map:
            not_applicable_permissions.append(permission)
    return not_applicable_permissions