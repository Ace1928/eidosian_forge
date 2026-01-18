from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.command_lib.iam import iam_util
def _AddActiveProjectFilterIfNotSpecified(filter_expr):
    if not filter_expr:
        return 'lifecycleState:ACTIVE'
    if 'lifecycleState' in filter_expr:
        return filter_expr
    return 'lifecycleState:ACTIVE AND ({})'.format(filter_expr)