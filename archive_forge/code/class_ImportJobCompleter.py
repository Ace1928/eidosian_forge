from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util import parameter_info_lib
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ImportJobCompleter(ListCommandCompleter):

    def __init__(self, **kwargs):
        super(ImportJobCompleter, self).__init__(collection=IMPORT_JOB_COLLECTION, list_command='beta kms import-jobs list --uri', flags=['location', 'keyring'], **kwargs)