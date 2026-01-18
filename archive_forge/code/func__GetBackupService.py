from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def _GetBackupService():
    """Returns the service for interacting with the Firestore Backup service."""
    return api_utils.GetClient().projects_locations_backups