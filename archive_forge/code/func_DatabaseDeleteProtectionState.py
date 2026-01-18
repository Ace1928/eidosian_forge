from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.firestore import flags
from googlecloudsdk.core import properties
def DatabaseDeleteProtectionState(self, enable_delete_protection):
    if enable_delete_protection:
        return api_utils.GetMessages().GoogleFirestoreAdminV1Database.DeleteProtectionStateValueValuesEnum.DELETE_PROTECTION_ENABLED
    return api_utils.GetMessages().GoogleFirestoreAdminV1Database.DeleteProtectionStateValueValuesEnum.DELETE_PROTECTION_DISABLED