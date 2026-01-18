from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.monitoring import completers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core.util import times
def AddCombinerFlag(parser, resource):
    """Adds flags for specifying a combiner, which defines how to combine the results of multiple conditions."""
    parser.add_argument('--combiner', choices={'COMBINE_UNSPECIFIED': 'An unspecified combiner', 'AND': 'An incident is created only if all conditions are met simultaneously. This combiner is satisfied if all conditions are met, even if they are met on completely different resources.', 'OR': 'An incident is created if any of the listed conditions is met.', 'AND_WITH_MATCHING_RESOURCE': 'Combine conditions using logical AND operator, but unlike the regular AND option, an incident is created only if all conditions are met simultaneously on at least one resource.'}, help='The combiner for the {}.'.format(resource))