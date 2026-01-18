from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddForceDeleteFlag(parser):
    """Adds a --force-delete flag to the given parser."""
    help_text = 'If set, the following restrictions against deletion of the backup vault instance can be overridden: * deletion of a backup vault instance containing no backups,but still contains empty datasources. * deletion of a backup vault instance containing no backups,but still contains empty datasources.'
    parser.add_argument('--force-delete', action='store_true', help=help_text)