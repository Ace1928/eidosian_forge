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
def CreateTable(self, reference, ignore_existing=False, schema=None, description=None, display_name=None, expiration=None, view_query=None, materialized_view_query=None, enable_refresh=None, refresh_interval_ms=None, max_staleness=None, external_data_config=None, biglake_config=None, view_udf_resources=None, use_legacy_sql=None, labels=None, time_partitioning=None, clustering=None, range_partitioning=None, require_partition_filter=None, destination_kms_key=None, location=None, table_constraints=None, resource_tags=None):
    """Create a table corresponding to TableReference.

    Args:
      reference: the TableReference to create.
      ignore_existing: (boolean, default False) If False, raise an exception if
        the dataset already exists.
      schema: an optional schema for tables.
      description: an optional description for tables or views.
      display_name: an optional friendly name for the table.
      expiration: optional expiration time in milliseconds since the epoch for
        tables or views.
      view_query: an optional Sql query for views.
      materialized_view_query: an optional standard SQL query for materialized
        views.
      enable_refresh: for materialized views, an optional toggle to enable /
        disable automatic refresh when the base table is updated.
      refresh_interval_ms: for materialized views, an optional maximum frequency
        for automatic refreshes.
      max_staleness: INTERVAL value that determines the maximum staleness
        allowed when querying a materialized view or an external table. By
        default no staleness is allowed.
      external_data_config: defines a set of external resources used to create
        an external table. For example, a BigQuery table backed by CSV files in
        GCS.
      biglake_config: specifies the configuration of a BigLake managed table.
      view_udf_resources: optional UDF resources used in a view.
      use_legacy_sql: The choice of using Legacy SQL for the query is optional.
        If not specified, the server will automatically determine the dialect
        based on query information, such as dialect prefixes. If no prefixes are
        found, it will default to Legacy SQL.
      labels: an optional dict of labels to set on the table.
      time_partitioning: if set, enables time based partitioning on the table
        and configures the partitioning.
      clustering: if set, enables and configures clustering on the table.
      range_partitioning: if set, enables range partitioning on the table and
        configures the partitioning.
      require_partition_filter: if set, partition filter is required for
        queiries over this table.
      destination_kms_key: User specified KMS key for encryption.
      location: an optional location for which to create tables or views.
      table_constraints: an optional primary key and foreign key configuration
        for the table.
      resource_tags: an optional dict of tags to attach to the table.

    Raises:
      TypeError: if reference is not a TableReference.
      BigqueryDuplicateError: if reference exists and ignore_existing
        is False.
    """
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.TableReference, method='CreateTable')
    try:
        body = bq_processor_utils.ConstructObjectInfo(reference)
        if schema is not None:
            body['schema'] = {'fields': schema}
        if display_name is not None:
            body['friendlyName'] = display_name
        if description is not None:
            body['description'] = description
        if expiration is not None:
            body['expirationTime'] = expiration
        if view_query is not None:
            view_args = {'query': view_query}
            if view_udf_resources is not None:
                view_args['userDefinedFunctionResources'] = view_udf_resources
            body['view'] = view_args
            if use_legacy_sql is not None:
                view_args['useLegacySql'] = use_legacy_sql
        if materialized_view_query is not None:
            materialized_view_args = {'query': materialized_view_query}
            if enable_refresh is not None:
                materialized_view_args['enableRefresh'] = enable_refresh
            if refresh_interval_ms is not None:
                materialized_view_args['refreshIntervalMs'] = refresh_interval_ms
            body['materializedView'] = materialized_view_args
        if external_data_config is not None:
            if max_staleness is not None:
                body['maxStaleness'] = max_staleness
            body['externalDataConfiguration'] = external_data_config
        if biglake_config is not None:
            body['biglakeConfiguration'] = biglake_config
        if labels is not None:
            body['labels'] = labels
        if time_partitioning is not None:
            body['timePartitioning'] = time_partitioning
        if clustering is not None:
            body['clustering'] = clustering
        if range_partitioning is not None:
            body['rangePartitioning'] = range_partitioning
        if require_partition_filter is not None:
            body['requirePartitionFilter'] = require_partition_filter
        if destination_kms_key is not None:
            body['encryptionConfiguration'] = {'kmsKeyName': destination_kms_key}
        if location is not None:
            body['location'] = location
        if table_constraints is not None:
            body['table_constraints'] = table_constraints
        if resource_tags is not None:
            body['resourceTags'] = resource_tags
        self.apiclient.tables().insert(body=body, **dict(reference.GetDatasetReference())).execute()
    except bq_error.BigqueryDuplicateError:
        if not ignore_existing:
            raise