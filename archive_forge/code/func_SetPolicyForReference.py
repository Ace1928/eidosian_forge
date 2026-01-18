from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from typing import Optional
from absl import app
from absl import flags
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from utils import bq_id_utils
def SetPolicyForReference(self, client, reference, policy):
    """Set the IAM policy for a table or dataset.

    Args:
      reference: A DatasetReference or TableReference.
      policy: The policy object, composed of dictionaries, lists, and primitive
        types.

    Raises:
      RuntimeError: reference isn't an expected type.
    """
    if isinstance(reference, bq_id_utils.ApiClientHelper.TableReference):
        return client.SetTableIAMPolicy(reference, policy)
    elif isinstance(reference, bq_id_utils.ApiClientHelper.DatasetReference):
        return client.SetDatasetIAMPolicy(reference, policy)
    elif isinstance(reference, bq_id_utils.ApiClientHelper.ConnectionReference):
        return client.SetConnectionIAMPolicy(reference, policy)
    raise RuntimeError('Unexpected reference type: {r_type}'.format(r_type=type(reference)))