from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dns import dns_keys
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
import six
def AddDescribeFlags(parser, hide_short_zone_flag=False, is_beta=False):
    flags.GetZoneArg('The name of the managed-zone the DNSKEY record belongs to', hide_short_zone_flag=hide_short_zone_flag).AddToParser(parser)
    flags.GetKeyArg(is_beta=is_beta).AddToParser(parser)
    parser.display_info.AddTransforms(GetTransforms())