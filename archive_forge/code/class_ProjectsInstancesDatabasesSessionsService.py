from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesDatabasesSessionsService(base_api.BaseApiService):
    """Service class for the projects_instances_databases_sessions resource."""
    _NAME = 'projects_instances_databases_sessions'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesDatabasesSessionsService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Creates multiple new sessions. This API can be used to initialize a session cache on the clients. See https://goo.gl/TgSFN2 for best practices on session cache management.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchCreateSessionsResponse) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions:batchCreate', http_method='POST', method_id='spanner.projects.instances.databases.sessions.batchCreate', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/sessions:batchCreate', request_field='batchCreateSessionsRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsBatchCreateRequest', response_type_name='BatchCreateSessionsResponse', supports_download=False)

    def BatchWrite(self, request, global_params=None):
        """Batches the supplied mutation groups in a collection of efficient transactions. All mutations in a group are committed atomically. However, mutations across groups can be committed non-atomically in an unspecified order and thus, they must be independent of each other. Partial failure is possible, i.e., some groups may have been committed successfully, while some may have failed. The results of individual batches are streamed into the response as the batches are applied. BatchWrite requests are not replay protected, meaning that each mutation group may be applied more than once. Replays of non-idempotent mutations may have undesirable effects. For example, replays of an insert mutation may produce an already exists error or if you use generated or commit timestamp-based keys, it may result in additional rows being added to the mutation's table. We recommend structuring your mutation groups to be idempotent to avoid this issue.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsBatchWriteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchWriteResponse) The response message.
      """
        config = self.GetMethodConfig('BatchWrite')
        return self._RunMethod(config, request, global_params=global_params)
    BatchWrite.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:batchWrite', http_method='POST', method_id='spanner.projects.instances.databases.sessions.batchWrite', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:batchWrite', request_field='batchWriteRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsBatchWriteRequest', response_type_name='BatchWriteResponse', supports_download=False)

    def BeginTransaction(self, request, global_params=None):
        """Begins a new transaction. This step can often be skipped: Read, ExecuteSql and Commit can begin a new transaction as a side-effect.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsBeginTransactionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Transaction) The response message.
      """
        config = self.GetMethodConfig('BeginTransaction')
        return self._RunMethod(config, request, global_params=global_params)
    BeginTransaction.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:beginTransaction', http_method='POST', method_id='spanner.projects.instances.databases.sessions.beginTransaction', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:beginTransaction', request_field='beginTransactionRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsBeginTransactionRequest', response_type_name='Transaction', supports_download=False)

    def Commit(self, request, global_params=None):
        """Commits a transaction. The request includes the mutations to be applied to rows in the database. `Commit` might return an `ABORTED` error. This can occur at any time; commonly, the cause is conflicts with concurrent transactions. However, it can also happen for a variety of other reasons. If `Commit` returns `ABORTED`, the caller should re-attempt the transaction from the beginning, re-using the same session. On very rare occasions, `Commit` might return `UNKNOWN`. This can happen, for example, if the client job experiences a 1+ hour networking failure. At that point, Cloud Spanner has lost track of the transaction outcome and we recommend that you perform another read from the database to see the state of things as they are now.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsCommitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CommitResponse) The response message.
      """
        config = self.GetMethodConfig('Commit')
        return self._RunMethod(config, request, global_params=global_params)
    Commit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:commit', http_method='POST', method_id='spanner.projects.instances.databases.sessions.commit', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:commit', request_field='commitRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsCommitRequest', response_type_name='CommitResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new session. A session can be used to perform transactions that read and/or modify data in a Cloud Spanner database. Sessions are meant to be reused for many consecutive transactions. Sessions can only execute one transaction at a time. To execute multiple concurrent read-write/write-only transactions, create multiple sessions. Note that standalone reads and queries use a transaction internally, and count toward the one transaction limit. Active sessions use additional server resources, so it is a good idea to delete idle and unneeded sessions. Aside from explicit deletes, Cloud Spanner may delete sessions for which no operations are sent for more than an hour. If a session is deleted, requests to it return `NOT_FOUND`. Idle sessions can be kept alive by sending a trivial SQL query periodically, e.g., `"SELECT 1"`.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Session) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions', http_method='POST', method_id='spanner.projects.instances.databases.sessions.create', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/sessions', request_field='createSessionRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsCreateRequest', response_type_name='Session', supports_download=False)

    def Delete(self, request, global_params=None):
        """Ends a session, releasing server resources associated with it. This will asynchronously trigger cancellation of any operations that are running with this session.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}', http_method='DELETE', method_id='spanner.projects.instances.databases.sessions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesSessionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def ExecuteBatchDml(self, request, global_params=None):
        """Executes a batch of SQL DML statements. This method allows many statements to be run with lower latency than submitting them sequentially with ExecuteSql. Statements are executed in sequential order. A request can succeed even if a statement fails. The ExecuteBatchDmlResponse.status field in the response provides information about the statement that failed. Clients must inspect this field to determine whether an error occurred. Execution stops after the first failed statement; the remaining statements are not executed.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsExecuteBatchDmlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExecuteBatchDmlResponse) The response message.
      """
        config = self.GetMethodConfig('ExecuteBatchDml')
        return self._RunMethod(config, request, global_params=global_params)
    ExecuteBatchDml.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:executeBatchDml', http_method='POST', method_id='spanner.projects.instances.databases.sessions.executeBatchDml', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:executeBatchDml', request_field='executeBatchDmlRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsExecuteBatchDmlRequest', response_type_name='ExecuteBatchDmlResponse', supports_download=False)

    def ExecuteSql(self, request, global_params=None):
        """Executes an SQL statement, returning all results in a single reply. This method cannot be used to return a result set larger than 10 MiB; if the query yields more data than that, the query fails with a `FAILED_PRECONDITION` error. Operations inside read-write transactions might return `ABORTED`. If this occurs, the application should restart the transaction from the beginning. See Transaction for more details. Larger result sets can be fetched in streaming fashion by calling ExecuteStreamingSql instead.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsExecuteSqlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResultSet) The response message.
      """
        config = self.GetMethodConfig('ExecuteSql')
        return self._RunMethod(config, request, global_params=global_params)
    ExecuteSql.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:executeSql', http_method='POST', method_id='spanner.projects.instances.databases.sessions.executeSql', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:executeSql', request_field='executeSqlRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsExecuteSqlRequest', response_type_name='ResultSet', supports_download=False)

    def ExecuteStreamingSql(self, request, global_params=None):
        """Like ExecuteSql, except returns the result set as a stream. Unlike ExecuteSql, there is no limit on the size of the returned result set. However, no individual row in the result set can exceed 100 MiB, and no column value can exceed 10 MiB.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsExecuteStreamingSqlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartialResultSet) The response message.
      """
        config = self.GetMethodConfig('ExecuteStreamingSql')
        return self._RunMethod(config, request, global_params=global_params)
    ExecuteStreamingSql.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:executeStreamingSql', http_method='POST', method_id='spanner.projects.instances.databases.sessions.executeStreamingSql', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:executeStreamingSql', request_field='executeSqlRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsExecuteStreamingSqlRequest', response_type_name='PartialResultSet', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a session. Returns `NOT_FOUND` if the session does not exist. This is mainly useful for determining whether a session is still alive.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Session) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}', http_method='GET', method_id='spanner.projects.instances.databases.sessions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesSessionsGetRequest', response_type_name='Session', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all sessions in a given database.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSessionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions', http_method='GET', method_id='spanner.projects.instances.databases.sessions.list', ordered_params=['database'], path_params=['database'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+database}/sessions', request_field='', request_type_name='SpannerProjectsInstancesDatabasesSessionsListRequest', response_type_name='ListSessionsResponse', supports_download=False)

    def PartitionQuery(self, request, global_params=None):
        """Creates a set of partition tokens that can be used to execute a query operation in parallel. Each of the returned partition tokens can be used by ExecuteStreamingSql to specify a subset of the query result to read. The same session and read-only transaction must be used by the PartitionQueryRequest used to create the partition tokens and the ExecuteSqlRequests that use the partition tokens. Partition tokens become invalid when the session used to create them is deleted, is idle for too long, begins a new transaction, or becomes too old. When any of these happen, it is not possible to resume the query, and the whole operation must be restarted from the beginning.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsPartitionQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartitionResponse) The response message.
      """
        config = self.GetMethodConfig('PartitionQuery')
        return self._RunMethod(config, request, global_params=global_params)
    PartitionQuery.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:partitionQuery', http_method='POST', method_id='spanner.projects.instances.databases.sessions.partitionQuery', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:partitionQuery', request_field='partitionQueryRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsPartitionQueryRequest', response_type_name='PartitionResponse', supports_download=False)

    def PartitionRead(self, request, global_params=None):
        """Creates a set of partition tokens that can be used to execute a read operation in parallel. Each of the returned partition tokens can be used by StreamingRead to specify a subset of the read result to read. The same session and read-only transaction must be used by the PartitionReadRequest used to create the partition tokens and the ReadRequests that use the partition tokens. There are no ordering guarantees on rows returned among the returned partition tokens, or even within each individual StreamingRead call issued with a partition_token. Partition tokens become invalid when the session used to create them is deleted, is idle for too long, begins a new transaction, or becomes too old. When any of these happen, it is not possible to resume the read, and the whole operation must be restarted from the beginning.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsPartitionReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartitionResponse) The response message.
      """
        config = self.GetMethodConfig('PartitionRead')
        return self._RunMethod(config, request, global_params=global_params)
    PartitionRead.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:partitionRead', http_method='POST', method_id='spanner.projects.instances.databases.sessions.partitionRead', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:partitionRead', request_field='partitionReadRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsPartitionReadRequest', response_type_name='PartitionResponse', supports_download=False)

    def Read(self, request, global_params=None):
        """Reads rows from the database using key lookups and scans, as a simple key/value style alternative to ExecuteSql. This method cannot be used to return a result set larger than 10 MiB; if the read matches more data than that, the read fails with a `FAILED_PRECONDITION` error. Reads inside read-write transactions might return `ABORTED`. If this occurs, the application should restart the transaction from the beginning. See Transaction for more details. Larger result sets can be yielded in streaming fashion by calling StreamingRead instead.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResultSet) The response message.
      """
        config = self.GetMethodConfig('Read')
        return self._RunMethod(config, request, global_params=global_params)
    Read.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:read', http_method='POST', method_id='spanner.projects.instances.databases.sessions.read', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:read', request_field='readRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsReadRequest', response_type_name='ResultSet', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Rolls back a transaction, releasing any locks it holds. It is a good idea to call this for any transaction that includes one or more Read or ExecuteSql requests and ultimately decides not to commit. `Rollback` returns `OK` if it successfully aborts the transaction, the transaction was already aborted, or the transaction is not found. `Rollback` never returns `ABORTED`.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:rollback', http_method='POST', method_id='spanner.projects.instances.databases.sessions.rollback', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:rollback', request_field='rollbackRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsRollbackRequest', response_type_name='Empty', supports_download=False)

    def StreamingRead(self, request, global_params=None):
        """Like Read, except returns the result set as a stream. Unlike Read, there is no limit on the size of the returned result set. However, no individual row in the result set can exceed 100 MiB, and no column value can exceed 10 MiB.

      Args:
        request: (SpannerProjectsInstancesDatabasesSessionsStreamingReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartialResultSet) The response message.
      """
        config = self.GetMethodConfig('StreamingRead')
        return self._RunMethod(config, request, global_params=global_params)
    StreamingRead.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/sessions/{sessionsId}:streamingRead', http_method='POST', method_id='spanner.projects.instances.databases.sessions.streamingRead', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:streamingRead', request_field='readRequest', request_type_name='SpannerProjectsInstancesDatabasesSessionsStreamingReadRequest', response_type_name='PartialResultSet', supports_download=False)