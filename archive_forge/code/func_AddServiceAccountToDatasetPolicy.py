from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.command_lib.iam import iam_util
def AddServiceAccountToDatasetPolicy(access_message_type, dataset_policy, member, role):
    """Add service account to dataset."""
    for entry in dataset_policy.access:
        if entry.role == role and member in entry.userByEmail:
            return False
    dataset_policy.access.append(access_message_type(userByEmail=member, role='{0}'.format(role)))
    return True