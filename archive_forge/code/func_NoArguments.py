from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
def NoArguments(self, args):
    """Determine if the user provided any flags indicating scope."""
    no_compute_args = args.zones is None and args.regions is None and (not getattr(args, 'global'))
    return no_compute_args