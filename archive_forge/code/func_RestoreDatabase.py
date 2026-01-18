from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def RestoreDatabase(project, destination_database, source_backup, source_database, snapshot_time):
    """Restores a Firestore database from either a backup or a snapshot.

  Args:
    project: the project ID to list databases, a string.
    destination_database: the database to restore to, a string.
    source_backup: the backup to restore from, a string.
    source_database: the source database which the snapshot belongs to, a
      string.
    snapshot_time: the version of source database to restore from, a string in
      google-datetime format.

  Returns:
    an Operation.
  """
    messages = api_utils.GetMessages()
    if source_backup:
        return _GetDatabaseService().Restore(messages.FirestoreProjectsDatabasesRestoreRequest(parent='projects/{}'.format(project), googleFirestoreAdminV1RestoreDatabaseRequest=messages.GoogleFirestoreAdminV1RestoreDatabaseRequest(backup=source_backup, databaseId=destination_database)))
    restore_from_snapshot = messages.GoogleFirestoreAdminV1DatabaseSnapshot(database=source_database, snapshotTime=snapshot_time)
    return _GetDatabaseService().Restore(messages.FirestoreProjectsDatabasesRestoreRequest(parent='projects/{}'.format(project), googleFirestoreAdminV1RestoreDatabaseRequest=messages.GoogleFirestoreAdminV1RestoreDatabaseRequest(databaseId=destination_database, databaseSnapshot=restore_from_snapshot)))