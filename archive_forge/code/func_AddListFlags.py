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
def AddListFlags(parser, hide_short_zone_flag=False):
    parser.display_info.AddFormat('table(id,keyTag,type,isActive,description)')
    base.URI_FLAG.RemoveFromParser(parser)
    base.PAGE_SIZE_FLAG.RemoveFromParser(parser)
    flags.GetZoneArg('The name of the managed-zone you want to list DNSKEY records for.', hide_short_zone_flag=hide_short_zone_flag).AddToParser(parser)
    parser.display_info.AddCacheUpdater(None)
    parser.display_info.AddTransforms(GetTransforms())