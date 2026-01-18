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
def UpdateTable(self, reference, schema=None, description=None, display_name=None, expiration=None, view_query=None, materialized_view_query=None, enable_refresh=None, refresh_interval_ms=None, max_staleness=None, external_data_config=None, view_udf_resources=None, use_legacy_sql=None, labels_to_set=None, label_keys_to_remove=None, time_partitioning=None, range_partitioning=None, clustering=None, require_partition_filter=None, etag=None, encryption_configuration=None, location=None, autodetect_schema=False, table_constraints=None, tags_to_attach: Optional[Dict[str, str]]=None, tags_to_remove: Optional[List[str]]=None, clear_all_tags: bool=False):
    """Updates a table.

    Args:
      reference: the TableReference to update.
      schema: an optional schema for tables.
      description: an optional description for tables or views.
      display_name: an optional friendly name for the table.
      expiration: optional expiration time in milliseconds since the epoch for
        tables or views. Specifying 0 removes expiration time.
      view_query: an optional Sql query to update a view.
      materialized_view_query: an optional Standard SQL query for materialized
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
      view_udf_resources: optional UDF resources used in a view.
      use_legacy_sql: The choice of using Legacy SQL for the query is optional.
        If not specified, the server will automatically determine the dialect
        based on query information, such as dialect prefixes. If no prefixes are
        found, it will default to Legacy SQL.
      labels_to_set: an optional dict of labels to set on this table.
      label_keys_to_remove: an optional list of label keys to remove from this
        table.
      time_partitioning: if set, enables time based partitioning on the table
        and configures the partitioning.
      range_partitioning: if set, enables range partitioning on the table and
        configures the partitioning.
      clustering: if set, enables clustering on the table and configures the
        clustering spec.
      require_partition_filter: if set, partition filter is required for
        queiries over this table.
      etag: if set, checks that etag in the existing table matches.
      encryption_configuration: Updates the encryption configuration.
      location: an optional location for which to update tables or views.
      autodetect_schema: an optional flag to perform autodetect of file schema.
      table_constraints: an optional primary key and foreign key configuration
        for the table.
      tags_to_attach: an optional dict of tags to attach to the table
      tags_to_remove: an optional list of tag keys to remove from the table
      clear_all_tags: if set, clears all the tags attached to the table
    Raises:
      TypeError: if reference is not a TableReference.
    """
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.TableReference, method='UpdateTable')
    existing_table = {}
    if clear_all_tags:
        existing_table = self._ExecuteGetTableRequest(reference)
    table = bq_processor_utils.ConstructObjectInfo(reference)
    maybe_skip_schema = False
    if schema is not None:
        table['schema'] = {'fields': schema}
    elif not maybe_skip_schema:
        table['schema'] = None
    if encryption_configuration is not None:
        table['encryptionConfiguration'] = encryption_configuration
    if display_name is not None:
        table['friendlyName'] = display_name
    if description is not None:
        table['description'] = description
    if expiration is not None:
        if expiration == 0:
            table['expirationTime'] = None
        else:
            table['expirationTime'] = expiration
    if view_query is not None:
        view_args = {'query': view_query}
        if view_udf_resources is not None:
            view_args['userDefinedFunctionResources'] = view_udf_resources
        if use_legacy_sql is not None:
            view_args['useLegacySql'] = use_legacy_sql
        table['view'] = view_args
    materialized_view_args = {}
    if materialized_view_query is not None:
        materialized_view_args['query'] = materialized_view_query
    if enable_refresh is not None:
        materialized_view_args['enableRefresh'] = enable_refresh
    if refresh_interval_ms is not None:
        materialized_view_args['refreshIntervalMs'] = refresh_interval_ms
    if materialized_view_args:
        table['materializedView'] = materialized_view_args
    if external_data_config is not None:
        table['externalDataConfiguration'] = external_data_config
        if max_staleness is not None:
            table['maxStaleness'] = max_staleness
    if 'labels' not in table:
        table['labels'] = {}
    table_labels = table['labels']
    if table_labels is None:
        raise ValueError('Missing labels in table.')
    if labels_to_set:
        for label_key, label_value in labels_to_set.items():
            table_labels[label_key] = label_value
    if label_keys_to_remove:
        for label_key in label_keys_to_remove:
            table_labels[label_key] = None
    if time_partitioning is not None:
        table['timePartitioning'] = time_partitioning
    if range_partitioning is not None:
        table['rangePartitioning'] = range_partitioning
    if clustering is not None:
        if clustering == {}:
            table['clustering'] = None
        else:
            table['clustering'] = clustering
    if require_partition_filter is not None:
        table['requirePartitionFilter'] = require_partition_filter
    if location is not None:
        table['location'] = location
    if table_constraints is not None:
        table['table_constraints'] = table_constraints
    resource_tags = {}
    if clear_all_tags and 'resourceTags' in existing_table:
        for tag in existing_table['resourceTags']:
            resource_tags[tag] = None
    else:
        for tag in tags_to_remove or []:
            resource_tags[tag] = None
    for tag in tags_to_attach or {}:
        resource_tags[tag] = tags_to_attach[tag]
    table['resourceTags'] = resource_tags
    self._ExecutePatchTableRequest(reference, table, autodetect_schema, etag)