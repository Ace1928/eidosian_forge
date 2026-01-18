from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from cloudsdk.google.protobuf import descriptor_pb2
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.core.util import files
def AddCommonListArgs(parser, additional_choices=None):
    """Add Common flags for the List operation group."""
    mutex_group = parser.add_group(mutex=True, required=True)
    mutex_group.add_argument('--instance-config', completer=InstanceConfigCompleter, help='The ID of the instance configuration the operation is executing on.')
    mutex_group.add_argument('--instance', completer=InstanceCompleter, help='The ID of the instance the operation is executing on.')
    Database(positional=False, required=False, text='For database operations, the name of the database the operations are executing on.').AddToParser(parser)
    Backup(positional=False, required=False, text='For backup operations, the name of the backup the operations are executing on.').AddToParser(parser)
    type_choices = {'INSTANCE': 'Returns instance operations for the given instance. Note, type=INSTANCE does not work with --database or --backup.', 'DATABASE': 'If only the instance is specified (--instance), returns all database operations associated with the databases in the instance. When a database is specified (--database), the command would return database operations for the given database.', 'BACKUP': 'If only the instance is specified (--instance), returns all backup operations associated with backups in the instance. When a backup is specified (--backup), only the backup operations for the given backup are returned.', 'DATABASE_RESTORE': 'Database restore operations are returned for all databases in the given instance (--instance only) or only those associated with the given database (--database)', 'DATABASE_CREATE': 'Database create operations are returned for all databases in the given instance (--instance only) or only those associated with the given database (--database)', 'DATABASE_UPDATE_DDL': 'Database update DDL operations are returned for all databases in the given instance (--instance only) or only those associated with the given database (--database)', 'INSTANCE_CONFIG_CREATE': 'Instance configuration create operations are returned for the given instance configuration (--instance-config).', 'INSTANCE_CONFIG_UPDATE': 'Instance configuration update operations are returned for the given instance configuration (--instance-config).'}
    if additional_choices is not None:
        type_choices.update(additional_choices)
    parser.add_argument('--type', default='', type=lambda x: x.upper(), choices=type_choices, help='(optional) List only the operations of the given type.')
    parser.display_info.AddFormat('\n          table(\n            name.basename():label=OPERATION_ID,\n            metadata.statements.join(sep="\n"),\n            done():label=DONE,\n            metadata.\'@type\'.split(\'.\').slice(-1:).join()\n          )\n        ')
    parser.display_info.AddCacheUpdater(None)
    parser.display_info.AddTransforms({'done': _TransformOperationDone})
    parser.display_info.AddTransforms({'database': _TransformDatabaseId})