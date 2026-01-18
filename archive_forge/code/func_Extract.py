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
def Extract(self, reference, destination_uris, print_header=None, field_delimiter=None, destination_format=None, trial_id=None, add_serving_default_signature=None, compression=None, use_avro_logical_types=None, **kwds):
    """Extract the given table from BigQuery.

    The job will execute synchronously if sync=True is provided as an
    argument or if self.sync is true.

    Args:
      reference: TableReference to read data from.
      destination_uris: String specifying one or more destination locations,
        separated by commas.
      print_header: Optional. Whether to print out a header row in the results.
      field_delimiter: Optional. Specifies the single byte field delimiter.
      destination_format: Optional. Format to extract table to. May be "CSV",
        "AVRO" or "NEWLINE_DELIMITED_JSON".
      trial_id: Optional. 1-based ID of the trial to be exported from a
        hyperparameter tuning model.
      add_serving_default_signature: Optional. Whether to add serving_default
        signature for BigQuery ML trained tf based models.
      compression: Optional. The compression type to use for exported files.
        Possible values include "GZIP" and "NONE". The default value is NONE.
      use_avro_logical_types: Optional. Whether to use avro logical types for
        applicable column types on extract jobs.
      **kwds: Passed on to self.ExecuteJob.

    Returns:
      The resulting job info.

    Raises:
      bq_error.BigqueryClientError: if required parameters are invalid.
    """
    bq_id_utils.typecheck(reference, (bq_id_utils.ApiClientHelper.TableReference, bq_id_utils.ApiClientHelper.ModelReference), method='Extract')
    uris = destination_uris.split(',')
    for uri in uris:
        if not uri.startswith(bq_processor_utils.GCS_SCHEME_PREFIX):
            raise bq_error.BigqueryClientError('Illegal URI: {}. Extract URI must start with "{}".'.format(uri, bq_processor_utils.GCS_SCHEME_PREFIX))
    if isinstance(reference, bq_id_utils.ApiClientHelper.TableReference):
        extract_config = {'sourceTable': dict(reference)}
    elif isinstance(reference, bq_id_utils.ApiClientHelper.ModelReference):
        extract_config = {'sourceModel': dict(reference)}
        if trial_id:
            extract_config.update({'modelExtractOptions': {'trialId': trial_id}})
        if add_serving_default_signature:
            extract_config.update({'modelExtractOptions': {'addServingDefaultSignature': add_serving_default_signature}})
    bq_processor_utils.ApplyParameters(extract_config, destination_uris=uris, destination_format=destination_format, print_header=print_header, field_delimiter=field_delimiter, compression=compression, use_avro_logical_types=use_avro_logical_types)
    return self.ExecuteJob(configuration={'extract': extract_config}, **kwds)