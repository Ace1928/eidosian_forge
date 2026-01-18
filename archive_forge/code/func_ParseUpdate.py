from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.backupdr import util
from googlecloudsdk.command_lib.backupdr import util as command_util
def ParseUpdate(self, description, effective_time, enforced_retention):
    updated_bv = self.messages.BackupVault()
    if description is not None:
        updated_bv.description = description
    if effective_time is not None:
        updated_bv.effectiveTime = effective_time
    if enforced_retention != 'Nones':
        updated_bv.enforcedRetentionDuration = enforced_retention
    return updated_bv