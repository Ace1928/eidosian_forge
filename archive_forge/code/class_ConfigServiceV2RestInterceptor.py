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
from google.api_core import operations_v1
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging_config
from google.longrunning import operations_pb2  # type: ignore
from .base import ConfigServiceV2Transport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class ConfigServiceV2RestInterceptor:
    """Interceptor for ConfigServiceV2.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the ConfigServiceV2RestTransport.

    .. code-block:: python
        class MyCustomConfigServiceV2Interceptor(ConfigServiceV2RestInterceptor):
            def pre_copy_log_entries(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_copy_log_entries(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_bucket(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_bucket_async(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_bucket_async(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_exclusion(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_exclusion(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_link(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_link(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_saved_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_saved_query(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_sink(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_sink(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_view(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_exclusion(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_link(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_delete_link(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_saved_query(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_sink(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_bucket(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_cmek_settings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_cmek_settings(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_exclusion(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_exclusion(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_link(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_link(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_settings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_settings(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_sink(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_sink(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_view(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_buckets(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_buckets(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_exclusions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_exclusions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_links(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_links(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_recent_queries(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_recent_queries(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_saved_queries(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_saved_queries(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_sinks(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_sinks(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_views(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_views(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_undelete_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_update_bucket(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_bucket(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_bucket_async(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_bucket_async(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_cmek_settings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_cmek_settings(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_exclusion(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_exclusion(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_settings(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_settings(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_sink(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_sink(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_view(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_view(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = ConfigServiceV2RestTransport(interceptor=MyCustomConfigServiceV2Interceptor())
        client = ConfigServiceV2Client(transport=transport)


    """

    def pre_copy_log_entries(self, request: logging_config.CopyLogEntriesRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CopyLogEntriesRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for copy_log_entries

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_copy_log_entries(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for copy_log_entries

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_create_bucket(self, request: logging_config.CreateBucketRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CreateBucketRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_bucket

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_create_bucket(self, response: logging_config.LogBucket) -> logging_config.LogBucket:
        """Post-rpc interceptor for create_bucket

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_create_bucket_async(self, request: logging_config.CreateBucketRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CreateBucketRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_bucket_async

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_create_bucket_async(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_bucket_async

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_create_exclusion(self, request: logging_config.CreateExclusionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CreateExclusionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_exclusion

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_create_exclusion(self, response: logging_config.LogExclusion) -> logging_config.LogExclusion:
        """Post-rpc interceptor for create_exclusion

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_create_link(self, request: logging_config.CreateLinkRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CreateLinkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_link

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_create_link(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for create_link

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_create_saved_query(self, request: logging_config.CreateSavedQueryRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CreateSavedQueryRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_saved_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_create_saved_query(self, response: logging_config.SavedQuery) -> logging_config.SavedQuery:
        """Post-rpc interceptor for create_saved_query

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_create_sink(self, request: logging_config.CreateSinkRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CreateSinkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_sink

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_create_sink(self, response: logging_config.LogSink) -> logging_config.LogSink:
        """Post-rpc interceptor for create_sink

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_create_view(self, request: logging_config.CreateViewRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.CreateViewRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_create_view(self, response: logging_config.LogView) -> logging_config.LogView:
        """Post-rpc interceptor for create_view

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_delete_bucket(self, request: logging_config.DeleteBucketRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.DeleteBucketRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_bucket

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def pre_delete_exclusion(self, request: logging_config.DeleteExclusionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.DeleteExclusionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_exclusion

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def pre_delete_link(self, request: logging_config.DeleteLinkRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.DeleteLinkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_link

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_delete_link(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for delete_link

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_delete_saved_query(self, request: logging_config.DeleteSavedQueryRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.DeleteSavedQueryRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_saved_query

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def pre_delete_sink(self, request: logging_config.DeleteSinkRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.DeleteSinkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_sink

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def pre_delete_view(self, request: logging_config.DeleteViewRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.DeleteViewRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def pre_get_bucket(self, request: logging_config.GetBucketRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.GetBucketRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_bucket

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_get_bucket(self, response: logging_config.LogBucket) -> logging_config.LogBucket:
        """Post-rpc interceptor for get_bucket

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_get_cmek_settings(self, request: logging_config.GetCmekSettingsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.GetCmekSettingsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_cmek_settings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_get_cmek_settings(self, response: logging_config.CmekSettings) -> logging_config.CmekSettings:
        """Post-rpc interceptor for get_cmek_settings

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_get_exclusion(self, request: logging_config.GetExclusionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.GetExclusionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_exclusion

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_get_exclusion(self, response: logging_config.LogExclusion) -> logging_config.LogExclusion:
        """Post-rpc interceptor for get_exclusion

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_get_link(self, request: logging_config.GetLinkRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.GetLinkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_link

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_get_link(self, response: logging_config.Link) -> logging_config.Link:
        """Post-rpc interceptor for get_link

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_get_settings(self, request: logging_config.GetSettingsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.GetSettingsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_settings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_get_settings(self, response: logging_config.Settings) -> logging_config.Settings:
        """Post-rpc interceptor for get_settings

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_get_sink(self, request: logging_config.GetSinkRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.GetSinkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_sink

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_get_sink(self, response: logging_config.LogSink) -> logging_config.LogSink:
        """Post-rpc interceptor for get_sink

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_get_view(self, request: logging_config.GetViewRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.GetViewRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_get_view(self, response: logging_config.LogView) -> logging_config.LogView:
        """Post-rpc interceptor for get_view

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_buckets(self, request: logging_config.ListBucketsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.ListBucketsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_buckets

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_list_buckets(self, response: logging_config.ListBucketsResponse) -> logging_config.ListBucketsResponse:
        """Post-rpc interceptor for list_buckets

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_exclusions(self, request: logging_config.ListExclusionsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.ListExclusionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_exclusions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_list_exclusions(self, response: logging_config.ListExclusionsResponse) -> logging_config.ListExclusionsResponse:
        """Post-rpc interceptor for list_exclusions

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_links(self, request: logging_config.ListLinksRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.ListLinksRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_links

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_list_links(self, response: logging_config.ListLinksResponse) -> logging_config.ListLinksResponse:
        """Post-rpc interceptor for list_links

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_recent_queries(self, request: logging_config.ListRecentQueriesRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.ListRecentQueriesRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_recent_queries

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_list_recent_queries(self, response: logging_config.ListRecentQueriesResponse) -> logging_config.ListRecentQueriesResponse:
        """Post-rpc interceptor for list_recent_queries

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_saved_queries(self, request: logging_config.ListSavedQueriesRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.ListSavedQueriesRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_saved_queries

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_list_saved_queries(self, response: logging_config.ListSavedQueriesResponse) -> logging_config.ListSavedQueriesResponse:
        """Post-rpc interceptor for list_saved_queries

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_sinks(self, request: logging_config.ListSinksRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.ListSinksRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_sinks

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_list_sinks(self, response: logging_config.ListSinksResponse) -> logging_config.ListSinksResponse:
        """Post-rpc interceptor for list_sinks

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_list_views(self, request: logging_config.ListViewsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.ListViewsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_views

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_list_views(self, response: logging_config.ListViewsResponse) -> logging_config.ListViewsResponse:
        """Post-rpc interceptor for list_views

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_undelete_bucket(self, request: logging_config.UndeleteBucketRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UndeleteBucketRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for undelete_bucket

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def pre_update_bucket(self, request: logging_config.UpdateBucketRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UpdateBucketRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_bucket

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_update_bucket(self, response: logging_config.LogBucket) -> logging_config.LogBucket:
        """Post-rpc interceptor for update_bucket

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_update_bucket_async(self, request: logging_config.UpdateBucketRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UpdateBucketRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_bucket_async

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_update_bucket_async(self, response: operations_pb2.Operation) -> operations_pb2.Operation:
        """Post-rpc interceptor for update_bucket_async

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_update_cmek_settings(self, request: logging_config.UpdateCmekSettingsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UpdateCmekSettingsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_cmek_settings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_update_cmek_settings(self, response: logging_config.CmekSettings) -> logging_config.CmekSettings:
        """Post-rpc interceptor for update_cmek_settings

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_update_exclusion(self, request: logging_config.UpdateExclusionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UpdateExclusionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_exclusion

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_update_exclusion(self, response: logging_config.LogExclusion) -> logging_config.LogExclusion:
        """Post-rpc interceptor for update_exclusion

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_update_settings(self, request: logging_config.UpdateSettingsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UpdateSettingsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_settings

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_update_settings(self, response: logging_config.Settings) -> logging_config.Settings:
        """Post-rpc interceptor for update_settings

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_update_sink(self, request: logging_config.UpdateSinkRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UpdateSinkRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_sink

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_update_sink(self, response: logging_config.LogSink) -> logging_config.LogSink:
        """Post-rpc interceptor for update_sink

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response

    def pre_update_view(self, request: logging_config.UpdateViewRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[logging_config.UpdateViewRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_view

        Override in a subclass to manipulate the request or metadata
        before they are sent to the ConfigServiceV2 server.
        """
        return (request, metadata)

    def post_update_view(self, response: logging_config.LogView) -> logging_config.LogView:
        """Post-rpc interceptor for update_view

        Override in a subclass to manipulate the response
        after it is returned by the ConfigServiceV2 server but before
        it is returned to user code.
        """
        return response