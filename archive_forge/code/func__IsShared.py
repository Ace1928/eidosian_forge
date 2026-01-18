from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.sole_tenancy.node_groups import flags
def _IsShared(share_setting):
    """"Transforms share settings to simple share settings information."""
    if share_setting and share_setting['shareType'] != 'LOCAL':
        return 'true'
    return 'false'