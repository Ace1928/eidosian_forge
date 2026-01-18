from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.firestore import flags
from googlecloudsdk.core import properties
def DatabaseType(self, database_type):
    if database_type == 'firestore-native':
        return api_utils.GetMessages().GoogleFirestoreAdminV1Database.TypeValueValuesEnum.FIRESTORE_NATIVE
    elif database_type == 'datastore-mode':
        return api_utils.GetMessages().GoogleFirestoreAdminV1Database.TypeValueValuesEnum.DATASTORE_MODE
    else:
        raise ValueError('invalid database type: {}'.format(database_type))