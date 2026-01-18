from google.auth.transport.requests import AuthorizedSession  # type: ignore
import json  # type: ignore
import grpc  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import path_template
from google.api_core import gapic_v1
from cloudsdk.google.protobuf import json_format
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import commit_response
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import result_set
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import spanner
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import transaction
from .base import SpannerTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class SpannerRestInterceptor:
    """Interceptor for Spanner.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the SpannerRestTransport.

    .. code-block:: python
        class MyCustomSpannerInterceptor(SpannerRestInterceptor):
            def pre_batch_create_sessions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_batch_create_sessions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_batch_write(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_batch_write(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_begin_transaction(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_begin_transaction(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_commit(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_commit(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_session(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_session(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_session(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_execute_batch_dml(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_execute_batch_dml(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_execute_sql(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_execute_sql(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_execute_streaming_sql(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_execute_streaming_sql(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_session(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_session(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_sessions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_sessions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_partition_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_partition_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_partition_read(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_partition_read(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_read(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_read(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_rollback(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_streaming_read(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_streaming_read(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = SpannerRestTransport(interceptor=MyCustomSpannerInterceptor())
        client = SpannerClient(transport=transport)


    """

    def pre_batch_create_sessions(self, request: spanner.BatchCreateSessionsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.BatchCreateSessionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for batch_create_sessions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_batch_create_sessions(self, response: spanner.BatchCreateSessionsResponse) -> spanner.BatchCreateSessionsResponse:
        """Post-rpc interceptor for batch_create_sessions

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_batch_write(self, request: spanner.BatchWriteRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.BatchWriteRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for batch_write

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_batch_write(self, response: rest_streaming.ResponseIterator) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for batch_write

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_begin_transaction(self, request: spanner.BeginTransactionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.BeginTransactionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for begin_transaction

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_begin_transaction(self, response: transaction.Transaction) -> transaction.Transaction:
        """Post-rpc interceptor for begin_transaction

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_commit(self, request: spanner.CommitRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.CommitRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for commit

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_commit(self, response: commit_response.CommitResponse) -> commit_response.CommitResponse:
        """Post-rpc interceptor for commit

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_create_session(self, request: spanner.CreateSessionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.CreateSessionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_session

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_create_session(self, response: spanner.Session) -> spanner.Session:
        """Post-rpc interceptor for create_session

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_delete_session(self, request: spanner.DeleteSessionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.DeleteSessionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_session

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def pre_execute_batch_dml(self, request: spanner.ExecuteBatchDmlRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.ExecuteBatchDmlRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for execute_batch_dml

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_execute_batch_dml(self, response: spanner.ExecuteBatchDmlResponse) -> spanner.ExecuteBatchDmlResponse:
        """Post-rpc interceptor for execute_batch_dml

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_execute_sql(self, request: spanner.ExecuteSqlRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.ExecuteSqlRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for execute_sql

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_execute_sql(self, response: result_set.ResultSet) -> result_set.ResultSet:
        """Post-rpc interceptor for execute_sql

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_execute_streaming_sql(self, request: spanner.ExecuteSqlRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.ExecuteSqlRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for execute_streaming_sql

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_execute_streaming_sql(self, response: rest_streaming.ResponseIterator) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for execute_streaming_sql

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_get_session(self, request: spanner.GetSessionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.GetSessionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_session

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_get_session(self, response: spanner.Session) -> spanner.Session:
        """Post-rpc interceptor for get_session

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_list_sessions(self, request: spanner.ListSessionsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.ListSessionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_sessions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_list_sessions(self, response: spanner.ListSessionsResponse) -> spanner.ListSessionsResponse:
        """Post-rpc interceptor for list_sessions

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_partition_query(self, request: spanner.PartitionQueryRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.PartitionQueryRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for partition_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_partition_query(self, response: spanner.PartitionResponse) -> spanner.PartitionResponse:
        """Post-rpc interceptor for partition_query

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_partition_read(self, request: spanner.PartitionReadRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.PartitionReadRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for partition_read

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_partition_read(self, response: spanner.PartitionResponse) -> spanner.PartitionResponse:
        """Post-rpc interceptor for partition_read

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_read(self, request: spanner.ReadRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.ReadRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for read

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_read(self, response: result_set.ResultSet) -> result_set.ResultSet:
        """Post-rpc interceptor for read

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response

    def pre_rollback(self, request: spanner.RollbackRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.RollbackRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for rollback

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def pre_streaming_read(self, request: spanner.ReadRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[spanner.ReadRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for streaming_read

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Spanner server.
        """
        return (request, metadata)

    def post_streaming_read(self, response: rest_streaming.ResponseIterator) -> rest_streaming.ResponseIterator:
        """Post-rpc interceptor for streaming_read

        Override in a subclass to manipulate the response
        after it is returned by the Spanner server but before
        it is returned to user code.
        """
        return response