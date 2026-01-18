from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
class ProjectsDatabasesDocumentsService(base_api.BaseApiService):
    """Service class for the projects_databases_documents resource."""
    _NAME = 'projects_databases_documents'

    def __init__(self, client):
        super(FirestoreV1.ProjectsDatabasesDocumentsService, self).__init__(client)
        self._upload_configs = {}

    def BatchGet(self, request, global_params=None):
        """Gets multiple documents. Documents returned by this method are not guaranteed to be returned in the same order that they were requested.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsBatchGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchGetDocumentsResponse) The response message.
      """
        config = self.GetMethodConfig('BatchGet')
        return self._RunMethod(config, request, global_params=global_params)
    BatchGet.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents:batchGet', http_method='POST', method_id='firestore.projects.databases.documents.batchGet', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/documents:batchGet', request_field='batchGetDocumentsRequest', request_type_name='FirestoreProjectsDatabasesDocumentsBatchGetRequest', response_type_name='BatchGetDocumentsResponse', supports_download=False)

    def BatchWrite(self, request, global_params=None):
        """Applies a batch of write operations. The BatchWrite method does not apply the write operations atomically and can apply them out of order. Method does not allow more than one write per document. Each write succeeds or fails independently. See the BatchWriteResponse for the success status of each write. If you require an atomically applied set of writes, use Commit instead.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsBatchWriteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchWriteResponse) The response message.
      """
        config = self.GetMethodConfig('BatchWrite')
        return self._RunMethod(config, request, global_params=global_params)
    BatchWrite.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents:batchWrite', http_method='POST', method_id='firestore.projects.databases.documents.batchWrite', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/documents:batchWrite', request_field='batchWriteRequest', request_type_name='FirestoreProjectsDatabasesDocumentsBatchWriteRequest', response_type_name='BatchWriteResponse', supports_download=False)

    def BeginTransaction(self, request, global_params=None):
        """Starts a new transaction.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsBeginTransactionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BeginTransactionResponse) The response message.
      """
        config = self.GetMethodConfig('BeginTransaction')
        return self._RunMethod(config, request, global_params=global_params)
    BeginTransaction.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents:beginTransaction', http_method='POST', method_id='firestore.projects.databases.documents.beginTransaction', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/documents:beginTransaction', request_field='beginTransactionRequest', request_type_name='FirestoreProjectsDatabasesDocumentsBeginTransactionRequest', response_type_name='BeginTransactionResponse', supports_download=False)

    def Commit(self, request, global_params=None):
        """Commits a transaction, while optionally updating documents.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsCommitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CommitResponse) The response message.
      """
        config = self.GetMethodConfig('Commit')
        return self._RunMethod(config, request, global_params=global_params)
    Commit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents:commit', http_method='POST', method_id='firestore.projects.databases.documents.commit', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/documents:commit', request_field='commitRequest', request_type_name='FirestoreProjectsDatabasesDocumentsCommitRequest', response_type_name='CommitResponse', supports_download=False)

    def CreateDocument(self, request, global_params=None):
        """Creates a new document.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsCreateDocumentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Document) The response message.
      """
        config = self.GetMethodConfig('CreateDocument')
        return self._RunMethod(config, request, global_params=global_params)
    CreateDocument.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{collectionId}', http_method='POST', method_id='firestore.projects.databases.documents.createDocument', ordered_params=['parent', 'collectionId'], path_params=['collectionId', 'parent'], query_params=['documentId', 'mask_fieldPaths'], relative_path='v1/{+parent}/{collectionId}', request_field='document', request_type_name='FirestoreProjectsDatabasesDocumentsCreateDocumentRequest', response_type_name='Document', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a document.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}', http_method='DELETE', method_id='firestore.projects.databases.documents.delete', ordered_params=['name'], path_params=['name'], query_params=['currentDocument_exists', 'currentDocument_updateTime'], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesDocumentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single document.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Document) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}', http_method='GET', method_id='firestore.projects.databases.documents.get', ordered_params=['name'], path_params=['name'], query_params=['mask_fieldPaths', 'readTime', 'transaction'], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesDocumentsGetRequest', response_type_name='Document', supports_download=False)

    def List(self, request, global_params=None):
        """Lists documents.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDocumentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}/{collectionId}', http_method='GET', method_id='firestore.projects.databases.documents.list', ordered_params=['parent', 'collectionId'], path_params=['collectionId', 'parent'], query_params=['mask_fieldPaths', 'orderBy', 'pageSize', 'pageToken', 'readTime', 'showMissing', 'transaction'], relative_path='v1/{+parent}/{collectionId}', request_field='', request_type_name='FirestoreProjectsDatabasesDocumentsListRequest', response_type_name='ListDocumentsResponse', supports_download=False)

    def ListCollectionIds(self, request, global_params=None):
        """Lists all the collection IDs underneath a document.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsListCollectionIdsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCollectionIdsResponse) The response message.
      """
        config = self.GetMethodConfig('ListCollectionIds')
        return self._RunMethod(config, request, global_params=global_params)
    ListCollectionIds.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}:listCollectionIds', http_method='POST', method_id='firestore.projects.databases.documents.listCollectionIds', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:listCollectionIds', request_field='listCollectionIdsRequest', request_type_name='FirestoreProjectsDatabasesDocumentsListCollectionIdsRequest', response_type_name='ListCollectionIdsResponse', supports_download=False)

    def ListDocuments(self, request, global_params=None):
        """Lists documents.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsListDocumentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDocumentsResponse) The response message.
      """
        config = self.GetMethodConfig('ListDocuments')
        return self._RunMethod(config, request, global_params=global_params)
    ListDocuments.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{collectionId}', http_method='GET', method_id='firestore.projects.databases.documents.listDocuments', ordered_params=['parent', 'collectionId'], path_params=['collectionId', 'parent'], query_params=['mask_fieldPaths', 'orderBy', 'pageSize', 'pageToken', 'readTime', 'showMissing', 'transaction'], relative_path='v1/{+parent}/{collectionId}', request_field='', request_type_name='FirestoreProjectsDatabasesDocumentsListDocumentsRequest', response_type_name='ListDocumentsResponse', supports_download=False)

    def Listen(self, request, global_params=None):
        """Listens to changes. This method is only available via gRPC or WebChannel (not REST).

      Args:
        request: (FirestoreProjectsDatabasesDocumentsListenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListenResponse) The response message.
      """
        config = self.GetMethodConfig('Listen')
        return self._RunMethod(config, request, global_params=global_params)
    Listen.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents:listen', http_method='POST', method_id='firestore.projects.databases.documents.listen', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/documents:listen', request_field='listenRequest', request_type_name='FirestoreProjectsDatabasesDocumentsListenRequest', response_type_name='ListenResponse', supports_download=False)

    def PartitionQuery(self, request, global_params=None):
        """Partitions a query by returning partition cursors that can be used to run the query in parallel. The returned partition cursors are split points that can be used by RunQuery as starting/end points for the query results.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsPartitionQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartitionQueryResponse) The response message.
      """
        config = self.GetMethodConfig('PartitionQuery')
        return self._RunMethod(config, request, global_params=global_params)
    PartitionQuery.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}:partitionQuery', http_method='POST', method_id='firestore.projects.databases.documents.partitionQuery', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:partitionQuery', request_field='partitionQueryRequest', request_type_name='FirestoreProjectsDatabasesDocumentsPartitionQueryRequest', response_type_name='PartitionQueryResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates or inserts a document.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Document) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}', http_method='PATCH', method_id='firestore.projects.databases.documents.patch', ordered_params=['name'], path_params=['name'], query_params=['currentDocument_exists', 'currentDocument_updateTime', 'mask_fieldPaths', 'updateMask_fieldPaths'], relative_path='v1/{+name}', request_field='document', request_type_name='FirestoreProjectsDatabasesDocumentsPatchRequest', response_type_name='Document', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Rolls back a transaction.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents:rollback', http_method='POST', method_id='firestore.projects.databases.documents.rollback', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/documents:rollback', request_field='rollbackRequest', request_type_name='FirestoreProjectsDatabasesDocumentsRollbackRequest', response_type_name='Empty', supports_download=False)

    def RunAggregationQuery(self, request, global_params=None):
        """Runs an aggregation query. Rather than producing Document results like Firestore.RunQuery, this API allows running an aggregation to produce a series of AggregationResult server-side. High-Level Example: ``` -- Return the number of documents in table given a filter. SELECT COUNT(*) FROM ( SELECT * FROM k where a = true ); ```.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsRunAggregationQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RunAggregationQueryResponse) The response message.
      """
        config = self.GetMethodConfig('RunAggregationQuery')
        return self._RunMethod(config, request, global_params=global_params)
    RunAggregationQuery.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}:runAggregationQuery', http_method='POST', method_id='firestore.projects.databases.documents.runAggregationQuery', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:runAggregationQuery', request_field='runAggregationQueryRequest', request_type_name='FirestoreProjectsDatabasesDocumentsRunAggregationQueryRequest', response_type_name='RunAggregationQueryResponse', supports_download=False)

    def RunQuery(self, request, global_params=None):
        """Runs a query.

      Args:
        request: (FirestoreProjectsDatabasesDocumentsRunQueryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RunQueryResponse) The response message.
      """
        config = self.GetMethodConfig('RunQuery')
        return self._RunMethod(config, request, global_params=global_params)
    RunQuery.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents/{documentsId}/{documentsId1}:runQuery', http_method='POST', method_id='firestore.projects.databases.documents.runQuery', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:runQuery', request_field='runQueryRequest', request_type_name='FirestoreProjectsDatabasesDocumentsRunQueryRequest', response_type_name='RunQueryResponse', supports_download=False)

    def Write(self, request, global_params=None):
        """Streams batches of document updates and deletes, in order. This method is only available via gRPC or WebChannel (not REST).

      Args:
        request: (FirestoreProjectsDatabasesDocumentsWriteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WriteResponse) The response message.
      """
        config = self.GetMethodConfig('Write')
        return self._RunMethod(config, request, global_params=global_params)
    Write.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/documents:write', http_method='POST', method_id='firestore.projects.databases.documents.write', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/documents:write', request_field='writeRequest', request_type_name='FirestoreProjectsDatabasesDocumentsWriteRequest', response_type_name='WriteResponse', supports_download=False)