from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.network_security import flags
from googlecloudsdk.command_lib.network_security import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListReferences(base.ListCommand):
    """Lists References of an Organization Address Group."""
    _release_track = base.ReleaseTrack.GA
    detailed_help = {'EXAMPLES': '        To list References of address group named my-address-group.\n\n          $ {command} my-address-group\n        '}

    @classmethod
    def Args(cls, parser):
        flags.AddOrganizationAddressGroupToParser(cls._release_track, parser)
        flags.AddListReferencesFormat(parser)

    def Run(self, args):
        return util.ListOrganizationAddressGroupReferences(self._release_track, args)