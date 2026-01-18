from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
def _TransformPatchJobDisplayName(resource):
    """Returns the display name of a patch job."""
    max_len = 15
    if resource.get('displayName', ''):
        name = resource['displayName']
    elif resource.get('patchDeployment', ''):
        name = osconfig_command_utils.GetResourceName(resource['patchDeployment'])
    else:
        name = ''
    return name[:max_len] + '..' if len(name) > max_len else name