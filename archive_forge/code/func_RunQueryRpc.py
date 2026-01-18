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
def RunQueryRpc(self, query, dry_run=None, use_cache=None, preserve_nulls=None, request_id=None, maximum_bytes_billed=None, max_results=None, wait=sys.maxsize, min_completion_ratio=None, wait_printer_factory: Optional[Callable[[], WaitPrinter]]=None, max_single_wait=None, external_table_definitions_json=None, udf_resources=None, location=None, connection_properties=None, **kwds):
    """Executes the given query using the rpc-style query api.

    Args:
      query: Query to execute.
      dry_run: Optional. Indicates whether the query will only be validated and
        return processing statistics instead of actually running.
      use_cache: Optional. Whether to use the query cache. Caching is
        best-effort only and you should not make assumptions about whether or
        how long a query result will be cached.
      preserve_nulls: Optional. Indicates whether to preserve nulls in input
        data. Temporary flag; will be removed in a future version.
      request_id: Optional. Specifies the idempotency token for the request.
      maximum_bytes_billed: Optional. Upper limit on maximum bytes billed.
      max_results: Optional. Maximum number of results to return.
      wait: (optional, default maxint) Max wait time in seconds.
      min_completion_ratio: Optional. Specifies the minimum fraction of data
        that must be scanned before a query returns. This value should be
        between 0.0 and 1.0 inclusive.
      wait_printer_factory: (optional, defaults to self.wait_printer_factory)
        Returns a subclass of WaitPrinter that will be called after each job
        poll.
      max_single_wait: Optional. Maximum number of seconds to wait for each call
        to query() / getQueryResults().
      external_table_definitions_json: Json representation of external table
        definitions.
      udf_resources: Array of inline and remote UDF resources.
      location: Optional. The geographic location where the job should run.
      connection_properties: Optional. Connection properties to use when running
        the query, presented as a list of key/value pairs. A key of "time_zone"
        indicates that the query will be run with the default timezone
        corresponding to the value.
      **kwds: Passed directly to self.ExecuteSyncQuery.

    Raises:
      bq_error.BigqueryClientError: if no query is provided.
      StopIteration: if the query does not complete within wait seconds.
      bq_error.BigqueryError: if query fails.

    Returns:
      A tuple (schema fields, row results, execution metadata).
        For regular queries, the execution metadata dict contains
        the 'State' and 'status' elements that would be in a job result
        after FormatJobInfo().
        For dry run queries schema and rows are empty, the execution metadata
        dict contains statistics
    """
    if not self.sync:
        raise bq_error.BigqueryClientError('Running RPC-style query asynchronously is not supported')
    if not query:
        raise bq_error.BigqueryClientError('No query string provided')
    if request_id is not None and (not flags.FLAGS.jobs_query_use_request_id):
        raise bq_error.BigqueryClientError('request_id is not yet supported')
    if wait_printer_factory:
        printer = wait_printer_factory()
    else:
        printer = self.wait_printer_factory()
    start_time = time.time()
    elapsed_time = 0
    job_reference = None
    current_wait_ms = None
    while True:
        try:
            elapsed_time = 0 if job_reference is None else time.time() - start_time
            remaining_time = wait - elapsed_time
            if max_single_wait is not None:
                current_wait_ms = int(min(remaining_time, max_single_wait) * 1000)
                if current_wait_ms < 0:
                    current_wait_ms = sys.maxsize
            if remaining_time < 0:
                raise StopIteration('Wait timed out. Query not finished.')
            if job_reference is None:
                rows_to_read = max_results
                if self.max_rows_per_request is not None:
                    if rows_to_read is None:
                        rows_to_read = self.max_rows_per_request
                    else:
                        rows_to_read = min(self.max_rows_per_request, int(rows_to_read))
                result = self._StartQueryRpc(query=query, preserve_nulls=preserve_nulls, request_id=request_id, maximum_bytes_billed=maximum_bytes_billed, use_cache=use_cache, dry_run=dry_run, min_completion_ratio=min_completion_ratio, timeout_ms=current_wait_ms, max_results=rows_to_read, external_table_definitions_json=external_table_definitions_json, udf_resources=udf_resources, location=location, connection_properties=connection_properties, **kwds)
                if dry_run:
                    execution = dict(statistics=dict(query=dict(totalBytesProcessed=result['totalBytesProcessed'], cacheHit=result['cacheHit'])))
                    if 'schema' in result:
                        execution['statistics']['query']['schema'] = result['schema']
                    return ([], [], execution)
                if 'jobReference' in result:
                    job_reference = bq_id_utils.ApiClientHelper.JobReference.Create(**result['jobReference'])
            else:
                printer.Print(job_reference.jobId, elapsed_time, 'RUNNING')
                result = self.GetQueryResults(job_reference.jobId, max_results=max_results, timeout_ms=current_wait_ms, location=location)
            if result['jobComplete']:
                schema, rows = self.ReadSchemaAndJobRows(dict(job_reference) if job_reference else {}, start_row=0, max_rows=max_results, result_first_page=result)
                status = {}
                if 'errors' in result:
                    status['errors'] = result['errors']
                execution = {'State': 'SUCCESS', 'status': status, 'jobReference': job_reference}
                return (schema, rows, execution)
        except bq_error.BigqueryCommunicationError as e:
            logging.warning('Transient error during query: %s', e)
        except bq_error.BigqueryBackendError as e:
            logging.warning('Transient error during query: %s', e)