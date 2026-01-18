from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def UpdateDataset(self, reference: bq_id_utils.ApiClientHelper.DatasetReference, description: Optional[str]=None, display_name: Optional[str]=None, acl=None, default_table_expiration_ms=None, default_partition_expiration_ms=None, labels_to_set=None, label_keys_to_remove=None, etag=None, default_kms_key=None, max_time_travel_hours=None, storage_billing_model=None):
    """Updates a dataset.

    Args:
      reference: the DatasetReference to update.
      description: an optional dataset description.
      display_name: an optional friendly name for the dataset.
      acl: an optional ACL for the dataset, as a list of dicts.
      default_table_expiration_ms: optional number of milliseconds for the
        default expiration duration for new tables created in this dataset.
      default_partition_expiration_ms: optional number of milliseconds for the
        default partition expiration duration for new partitioned tables created
        in this dataset.
      labels_to_set: an optional dict of labels to set on this dataset.
      label_keys_to_remove: an optional list of label keys to remove from this
        dataset.
      etag: if set, checks that etag in the existing dataset matches.
      default_kms_key: An optional kms dey that will apply to all newly created
        tables in the dataset, if no explicit key is supplied in the creating
        request.
      max_time_travel_hours: Optional. Define the max time travel in hours. The
        value can be from 48 to 168 hours (2 to 7 days). The default value is
        168 hours if this is not set.
      storage_billing_model: Optional. Sets the storage billing model for the
        dataset.

    Raises:
      TypeError: if reference is not a DatasetReference.
    """
    return client_dataset.UpdateDataset(apiclient=self.apiclient, reference=reference, description=description, display_name=display_name, acl=acl, default_table_expiration_ms=default_table_expiration_ms, default_partition_expiration_ms=default_partition_expiration_ms, labels_to_set=labels_to_set, label_keys_to_remove=label_keys_to_remove, etag=etag, default_kms_key=default_kms_key, max_time_travel_hours=max_time_travel_hours, storage_billing_model=storage_billing_model)