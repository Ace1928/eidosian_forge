from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _ActiveIpsArgument(required=False):
    return compute_flags.ResourceArgument(name='--source-nat-active-ips', detailed_help=_ACTIVE_IPS_HELP, resource_name='address', regional_collection='compute.addresses', region_hidden=True, plural=True, required=required)