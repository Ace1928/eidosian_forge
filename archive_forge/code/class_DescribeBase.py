from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import dns_keys
from googlecloudsdk.core import properties
class DescribeBase(object):
    """Show details about a DNSKEY."""
    detailed_help = dns_keys.DESCRIBE_HELP

    @staticmethod
    def Args(parser):
        dns_keys.AddDescribeFlags(parser, is_beta=True)

    def Run(self, args):
        keys = dns_keys.Keys.FromApiVersion(self.GetApiVersion())
        return keys.Describe(args.key_id, zone=args.zone, project=properties.VALUES.core.project.GetOrFail)

    def GetApiVersion(self):
        raise NotImplementedError