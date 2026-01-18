from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddDeleteSubPrefixArgs(parser):
    """Adds flags for delegate sub prefixes delete command."""
    _AddCommonSubPrefixArgs(parser, 'delete')