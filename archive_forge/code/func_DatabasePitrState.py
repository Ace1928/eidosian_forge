from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.firestore import flags
from googlecloudsdk.core import properties
def DatabasePitrState(self, enable_pitr):
    if enable_pitr is None:
        return api_utils.GetMessages().GoogleFirestoreAdminV1Database.PointInTimeRecoveryEnablementValueValuesEnum.POINT_IN_TIME_RECOVERY_ENABLEMENT_UNSPECIFIED
    if enable_pitr:
        return api_utils.GetMessages().GoogleFirestoreAdminV1Database.PointInTimeRecoveryEnablementValueValuesEnum.POINT_IN_TIME_RECOVERY_ENABLED
    return api_utils.GetMessages().GoogleFirestoreAdminV1Database.PointInTimeRecoveryEnablementValueValuesEnum.POINT_IN_TIME_RECOVERY_DISABLED