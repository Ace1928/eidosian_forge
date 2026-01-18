from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddEffectiveTime(parser):
    """Adds the --effective-time flag to the given parser."""
    help_text = 'The time at which the enforced retention period becomes locked. It should be specified in the format of "YYYY-MM-DD".'
    parser.add_argument('--effective-time', type=arg_parsers.Day.Parse, help=help_text)