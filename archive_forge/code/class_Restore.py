from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Restore(base.Command):
    """Restores a Cloud Firestore database from a backup.

  ## EXAMPLES

  To restore a database from a backup.

      $ {command}
      --source-backup=projects/PROJECT_ID/locations/LOCATION_ID/backups/BACKUP_ID
      --destination-database=DATABASE_ID

  To restore a database from a database snapshot.

      $ {command}
      --source-database=SOURCE_DB --snapshot-time=2023-05-26T10:20:00.00Z
      --destination-database=DATABASE_ID
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('--destination-database', metavar='DESTINATION_DATABASE', type=str, required=True, help=textwrap.dedent('            Destination database to restore to. Destination database will be created in the same location as the source backup.\n\n            This value should be 4-63 characters. Valid characters are /[a-z][0-9]-/\n            with first character a letter and the last a letter or a number. Must\n            not be UUID-like /[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}/.\n\n            Using "(default)" database ID is also allowed.\n\n            For example, to restore to database `testdb`:\n\n            $ {command} --destination-database=testdb\n            '))
        restore_source = parser.add_mutually_exclusive_group(required=True)
        restore_source.add_argument('--source-backup', metavar='SOURCE_BACKUP', type=str, required=False, help=textwrap.dedent('            The source backup to restore from.\n\n            For example, to restore from backup `cf9f748a-7980-4703-b1a1-d1ffff591db0` in us-east1:\n\n            $ {command} --source-backup=projects/PROJECT_ID/locations/us-east1/backups/cf9f748a-7980-4703-b1a1-d1ffff591db0\n            '))
        restore_from_snapshot = restore_source.add_argument_group(required=False, hidden=True, help=textwrap.dedent('            The database snapshot to restore from.\n\n            For example, to restore from snapshot `2023-05-26T10:20:00.00Z` of source database `source-db`:\n\n            $ {command} --source-database=source-db --snapshot-time=2023-05-26T10:20:00.00Z\n            '))
        restore_from_snapshot.add_argument('--source-database', metavar='SOURCE_DATABASE', type=str, required=True, help=textwrap.dedent('            The source database which the snapshot belongs to.\n\n            For example, to restore from a snapshot which belongs to source database `source-db`:\n\n            $ {command} --source-database=source-db\n            '))
        restore_from_snapshot.add_argument('--snapshot-time', metavar='SNAPSHOT_TIME', required=True, help=textwrap.dedent("            The version of source database which will be restored from.\n            This timestamp must be in the past, rounded to the minute and not older than the source database's `earliestVersionTime`. Note that Point-in-time recovery(PITR) must be enabled in the source database to use this feature.\n\n            For example, to restore from database snapshot at `2023-05-26T10:20:00.00Z`:\n\n            $ {command} --snapshot-time=2023-05-26T10:20:00.00Z\n            "))

    def Run(self, args):
        project = properties.VALUES.core.project.Get(required=True)
        return databases.RestoreDatabase(project, args.destination_database, args.source_backup, args.source_database, args.snapshot_time)