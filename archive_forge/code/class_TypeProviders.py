from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class TypeProviders(base.Group):
    """Commands for Deployment Manager type providers."""
    detailed_help = {'EXAMPLES': '          To view the details of a type provider, run:\n\n            $ {command} describe TYPE_NAME\n\n          To see the list of all type providers, run:\n\n            $ {command} list\n\n          More information about type providers:\n          https://cloud.google.com/deployment-manager/docs/fundamentals#basetypes\n          '}