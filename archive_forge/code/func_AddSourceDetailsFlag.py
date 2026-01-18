from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.database_migration.connection_profiles import flags as cp_flags
def AddSourceDetailsFlag(parser):
    """Adds the source connection profile parameters to the given parser."""
    source_cp_params_group = parser.add_group()
    AddGcsBucket(source_cp_params_group)
    AddGcsPrefix(source_cp_params_group)
    AddProviderFlag(source_cp_params_group)