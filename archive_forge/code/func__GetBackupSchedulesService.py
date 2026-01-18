from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as ex
def _GetBackupSchedulesService():
    """Returns the service to interact with the Firestore Backup Schedules."""
    return api_utils.GetClient().projects_databases_backupSchedules