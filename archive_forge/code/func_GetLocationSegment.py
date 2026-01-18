from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.args import common_args
def GetLocationSegment(args):
    """Returns the resource name up to the location."""
    parent = GetResourceSegment(args)
    return '{}/locations/{}'.format(parent, args.location)