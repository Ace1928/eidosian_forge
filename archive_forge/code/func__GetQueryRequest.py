from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from apitools.base.py import http_wrapper
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
def _GetQueryRequest(sql, query_mode, session_ref=None, read_only_options=None, request_options=None, enable_partitioned_dml=False):
    """Formats the request based on whether the statement contains DML.

  Args:
    sql: String, The SQL to execute.
    query_mode: String, The mode in which to run the query. Must be one of
      'NORMAL', 'PLAN', or 'PROFILE'
    session_ref: Reference to the session.
    read_only_options: The ReadOnly message for a read-only request. It is
      ignored in a DML request.
    request_options: The RequestOptions message that contains the priority.
    enable_partitioned_dml: Boolean, whether partitioned dml is enabled.

  Returns:
    ExecuteSqlRequest parameters
  """
    msgs = apis.GetMessagesModule('spanner', 'v1')
    if enable_partitioned_dml is True:
        transaction = _GetPartitionedDmlTransaction(session_ref)
    elif QueryHasDml(sql):
        transaction_options = msgs.TransactionOptions(readWrite=msgs.ReadWrite())
        transaction = msgs.TransactionSelector(begin=transaction_options)
    else:
        transaction_options = msgs.TransactionOptions(readOnly=read_only_options)
        transaction = msgs.TransactionSelector(singleUse=transaction_options)
    return msgs.ExecuteSqlRequest(sql=sql, requestOptions=request_options, queryMode=msgs.ExecuteSqlRequest.QueryModeValueValuesEnum(query_mode), transaction=transaction)