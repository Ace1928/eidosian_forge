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
def CopyTable(self, source_references, dest_reference, create_disposition=None, write_disposition=None, ignore_already_exists=False, encryption_configuration=None, operation_type='COPY', destination_expiration_time=None, **kwds):
    """Copies a table.

    Args:
      source_references: TableReferences of source tables.
      dest_reference: TableReference of destination table.
      create_disposition: Optional. Specifies the create_disposition for the
        dest_reference.
      write_disposition: Optional. Specifies the write_disposition for the
        dest_reference.
      ignore_already_exists: Whether to ignore "already exists" errors.
      encryption_configuration: Optional. Allows user to encrypt the table from
        the copy table command with Cloud KMS key. Passed as a dictionary in the
        following format: {'kmsKeyName': 'destination_kms_key'}
      **kwds: Passed on to ExecuteJob.

    Returns:
      The job description, or None for ignored errors.

    Raises:
      BigqueryDuplicateError: when write_disposition 'WRITE_EMPTY' is
        specified and the dest_reference table already exists.
    """
    for src_ref in source_references:
        bq_id_utils.typecheck(src_ref, bq_id_utils.ApiClientHelper.TableReference, method='CopyTable')
    bq_id_utils.typecheck(dest_reference, bq_id_utils.ApiClientHelper.TableReference, method='CopyTable')
    copy_config = {'destinationTable': dict(dest_reference), 'sourceTables': [dict(src_ref) for src_ref in source_references]}
    if encryption_configuration:
        copy_config['destinationEncryptionConfiguration'] = encryption_configuration
    if operation_type:
        copy_config['operationType'] = operation_type
    if destination_expiration_time:
        copy_config['destinationExpirationTime'] = destination_expiration_time
    bq_processor_utils.ApplyParameters(copy_config, create_disposition=create_disposition, write_disposition=write_disposition)
    try:
        return self.ExecuteJob({'copy': copy_config}, **kwds)
    except bq_error.BigqueryDuplicateError as e:
        if ignore_already_exists:
            return None
        raise e