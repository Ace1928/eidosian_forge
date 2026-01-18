from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def GetRegionalizedResourceName(args, version):
    """Returns regionalized resource name."""
    location = scc_util.ValidateAndGetLocation(args, version)
    name_components = args.finding.split('/')
    return f'{name_components[0]}/{name_components[1]}/{name_components[2]}/{name_components[3]}/locations/{location}/{name_components[4]}/{name_components[5]}'