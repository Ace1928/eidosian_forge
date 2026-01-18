from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from absl import flags
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
Perform an extract operation of source into destination_uris.

    Usage:
      extract <source_table> <destination_uris>

    Use -m option to extract a source_model.

    Examples:
      bq extract ds.table gs://mybucket/table.csv
      bq extract -m ds.model gs://mybucket/model

    Arguments:
      source_table: Source table to extract.
      source_model: Source model to extract.
      destination_uris: One or more Google Cloud Storage URIs, separated by
        commas.
    