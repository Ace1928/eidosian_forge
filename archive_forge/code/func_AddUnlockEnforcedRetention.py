from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddUnlockEnforcedRetention(parser):
    """Adds the --unlock-enforced-retention flag to the given parser."""
    help_text = 'Removes the lock on the enforced retention period, and resets the effective time. When unlocked, the enforced retention period can be changed at any time.'
    parser.add_argument('--unlock-enforced-retention', action='store_true', help=help_text)