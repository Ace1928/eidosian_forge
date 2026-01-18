from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddRegionFlag(parser, operation):
    """Adds the region argument to the argparse to specify the security policy region."""
    return compute_flags.AddRegionFlag(parser, 'security policy', operation, explanation=compute_flags.REGION_PROPERTY_EXPLANATION_NO_DEFAULT)