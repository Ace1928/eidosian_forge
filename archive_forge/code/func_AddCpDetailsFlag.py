from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
def AddCpDetailsFlag(parser):
    """Adds the source and destination parameters to the given parser."""
    cp_params_group = parser.add_group(required=True, mutex=True)
    AddSourceDetailsFlag(cp_params_group)
    cp_flags.AddCloudSQLInstanceFlag(cp_params_group)