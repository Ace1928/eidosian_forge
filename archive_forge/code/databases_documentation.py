from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
Restores a Firestore database from either a backup or a snapshot.

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
  