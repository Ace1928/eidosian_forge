from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
def _ParseUserType(alloydb_messages, user_type):
    if user_type == 'BUILT_IN':
        return alloydb_messages.User.UserTypeValueValuesEnum.ALLOYDB_BUILT_IN
    elif user_type == 'IAM_BASED':
        return alloydb_messages.User.UserTypeValueValuesEnum.ALLOYDB_IAM_USER
    return None