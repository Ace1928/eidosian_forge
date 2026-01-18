from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import dns_keys
from googlecloudsdk.core import properties
class ListBase(object):
    """View the list of all your DNSKEY records."""
    detailed_help = dns_keys.LIST_HELP

    @staticmethod
    def Args(parser):
        dns_keys.AddListFlags(parser)

    def Run(self, args):
        keys = dns_keys.Keys.FromApiVersion(self.GetApiVersion())
        return keys.List(args.zone, properties.VALUES.core.project.GetOrFail)

    def GetApiVersion(self):
        raise NotImplementedError