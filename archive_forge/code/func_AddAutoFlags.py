from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from surface.container.clusters import create
def AddAutoFlags(parser, release_track):
    """Adds flags that are not same in create."""
    flags.AddLoggingFlag(parser, True)
    flags.AddMonitoringFlag(parser, True)
    flags.AddBinauthzFlags(parser, release_track=release_track, autopilot=True)
    flags.AddWorkloadPoliciesFlag(parser)
    flags.AddReleaseChannelFlag(parser, autopilot=True)
    flags.AddEnableBackupRestoreFlag(parser)
    flags.AddAutoprovisioningResourceManagerTagsCreate(parser)
    flags.AddAdditiveVPCScopeFlags(parser, release_track=release_track)
    flags.AddIPAliasRelatedFlags(parser, autopilot=True)
    flags.AddEnableConfidentialNodesFlag(parser, hidden=True)