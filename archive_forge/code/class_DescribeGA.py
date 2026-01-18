from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import dns_keys
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DescribeGA(DescribeBase, base.DescribeCommand):

    @staticmethod
    def Args(parser):
        dns_keys.AddDescribeFlags(parser, hide_short_zone_flag=True)

    def GetApiVersion(self):
        return 'v1'