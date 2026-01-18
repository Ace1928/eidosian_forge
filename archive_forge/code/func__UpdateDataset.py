from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import time
from typing import Dict, List, Optional
from absl import app
from absl import flags
from pyglib import appcommands
import bq_utils
from clients import bigquery_client_extended
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from frontend import utils_data_transfer
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def _UpdateDataset(client: bigquery_client_extended.BigqueryClientExtended, reference: bq_id_utils.ApiClientHelper.DatasetReference, description: Optional[str]=None, source=None, default_table_expiration_ms=None, default_partition_expiration_ms=None, labels_to_set=None, label_keys_to_remove=None, etag=None, default_kms_key=None, max_time_travel_hours=None, storage_billing_model=None, tags_to_attach: Optional[Dict[str, str]]=None, tags_to_remove: Optional[List[str]]=None, clear_all_tags: Optional[bool]=None):
    """Updates a dataset.

  Reads JSON file if specified and loads updated values, before calling bigquery
  dataset update.

  Args:
    client: the BigQuery client.
    reference: the DatasetReference to update.
    description: an optional dataset description.
    source: an optional filename containing the JSON payload.
    default_table_expiration_ms: optional number of milliseconds for the default
      expiration duration for new tables created in this dataset.
    default_partition_expiration_ms: optional number of milliseconds for the
      default partition expiration duration for new partitioned tables created
      in this dataset.
    labels_to_set: an optional dict of labels to set on this dataset.
    label_keys_to_remove: an optional list of label keys to remove from this
      dataset.
    default_kms_key: an optional CMEK encryption key for all new tables in the
      dataset.
    max_time_travel_hours: Optional. Define the max time travel in hours. The
      value can be from 48 to 168 hours (2 to 7 days). The default value is 168
      hours if this is not set.
    storage_billing_model: Optional. Sets the storage billing model for the
      dataset.
    tags_to_attach: an optional dict of tags to attach to the dataset.
    tags_to_remove: an optional list of tag keys to remove from the dataset.
    clear_all_tags: if set, clears all the tags attached to the dataset.

  Raises:
    UsageError: when incorrect usage or invalid args are used.
  """
    acl = None
    if source is not None:
        if not os.path.exists(source):
            raise app.UsageError('Source file not found: %s' % (source,))
        if not os.path.isfile(source):
            raise app.UsageError('Source path is not a file: %s' % (source,))
        with open(source) as f:
            try:
                payload = json.load(f)
                if payload.__contains__('description'):
                    description = payload['description']
                if payload.__contains__('access'):
                    acl = payload['access']
            except ValueError as e:
                raise app.UsageError('Error decoding JSON schema from file %s: %s' % (source, e))
    client.UpdateDataset(reference, description=description, acl=acl, default_table_expiration_ms=default_table_expiration_ms, default_partition_expiration_ms=default_partition_expiration_ms, labels_to_set=labels_to_set, label_keys_to_remove=label_keys_to_remove, etag=etag, default_kms_key=default_kms_key, max_time_travel_hours=max_time_travel_hours, storage_billing_model=storage_billing_model)