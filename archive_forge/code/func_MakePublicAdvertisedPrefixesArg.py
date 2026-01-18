from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakePublicAdvertisedPrefixesArg():
    return compute_flags.ResourceArgument(resource_name='public advertised prefix', global_collection='compute.publicAdvertisedPrefixes')