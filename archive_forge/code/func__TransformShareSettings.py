from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.sole_tenancy.node_groups import flags
def _TransformShareSettings(share_setting):
    """"Transforms share settings to detailed share settings information."""
    if not share_setting or share_setting['shareType'] == 'LOCAL':
        return 'local'
    elif share_setting['shareType'] == 'SPECIFIC_PROJECTS':
        projects = share_setting['projectMap'] if 'projectMap' in share_setting else []
        return 'specific_project:' + ','.join(sorted(projects))
    elif share_setting['shareType'] == 'ORGANIZATION':
        return 'org'
    return ''