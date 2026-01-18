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
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings
from google.iam.v1 import iam_policy_pb2  # type: ignore
from google.iam.v1 import policy_pb2  # type: ignore
from cloudsdk.google.protobuf import empty_pb2  # type: ignore
from google.pubsub_v1.types import pubsub
from .base import SubscriberTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class SubscriberRestInterceptor:
    """Interceptor for Subscriber.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the SubscriberRestTransport.

    .. code-block:: python
        class MyCustomSubscriberInterceptor(SubscriberRestInterceptor):
            def pre_acknowledge(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_create_snapshot(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_snapshot(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_subscription(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_subscription(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_snapshot(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_subscription(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_snapshot(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_snapshot(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_subscription(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_subscription(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_snapshots(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_snapshots(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_subscriptions(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_subscriptions(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_modify_ack_deadline(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_modify_push_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_pull(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_pull(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_seek(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_seek(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_snapshot(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_snapshot(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_subscription(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_subscription(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = SubscriberRestTransport(interceptor=MyCustomSubscriberInterceptor())
        client = SubscriberClient(transport=transport)


    """

    def pre_acknowledge(self, request: pubsub.AcknowledgeRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.AcknowledgeRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for acknowledge

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def pre_create_snapshot(self, request: pubsub.CreateSnapshotRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.CreateSnapshotRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_snapshot

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_create_snapshot(self, response: pubsub.Snapshot) -> pubsub.Snapshot:
        """Post-rpc interceptor for create_snapshot

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_create_subscription(self, request: pubsub.Subscription, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.Subscription, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_subscription

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_create_subscription(self, response: pubsub.Subscription) -> pubsub.Subscription:
        """Post-rpc interceptor for create_subscription

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_delete_snapshot(self, request: pubsub.DeleteSnapshotRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.DeleteSnapshotRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_snapshot

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def pre_delete_subscription(self, request: pubsub.DeleteSubscriptionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.DeleteSubscriptionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_subscription

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def pre_get_snapshot(self, request: pubsub.GetSnapshotRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.GetSnapshotRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_snapshot

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_get_snapshot(self, response: pubsub.Snapshot) -> pubsub.Snapshot:
        """Post-rpc interceptor for get_snapshot

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_get_subscription(self, request: pubsub.GetSubscriptionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.GetSubscriptionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_subscription

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_get_subscription(self, response: pubsub.Subscription) -> pubsub.Subscription:
        """Post-rpc interceptor for get_subscription

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_list_snapshots(self, request: pubsub.ListSnapshotsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.ListSnapshotsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_snapshots

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_list_snapshots(self, response: pubsub.ListSnapshotsResponse) -> pubsub.ListSnapshotsResponse:
        """Post-rpc interceptor for list_snapshots

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_list_subscriptions(self, request: pubsub.ListSubscriptionsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.ListSubscriptionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_subscriptions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_list_subscriptions(self, response: pubsub.ListSubscriptionsResponse) -> pubsub.ListSubscriptionsResponse:
        """Post-rpc interceptor for list_subscriptions

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_modify_ack_deadline(self, request: pubsub.ModifyAckDeadlineRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.ModifyAckDeadlineRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for modify_ack_deadline

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def pre_modify_push_config(self, request: pubsub.ModifyPushConfigRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.ModifyPushConfigRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for modify_push_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def pre_pull(self, request: pubsub.PullRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.PullRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for pull

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_pull(self, response: pubsub.PullResponse) -> pubsub.PullResponse:
        """Post-rpc interceptor for pull

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_seek(self, request: pubsub.SeekRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.SeekRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for seek

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_seek(self, response: pubsub.SeekResponse) -> pubsub.SeekResponse:
        """Post-rpc interceptor for seek

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_update_snapshot(self, request: pubsub.UpdateSnapshotRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.UpdateSnapshotRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_snapshot

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_update_snapshot(self, response: pubsub.Snapshot) -> pubsub.Snapshot:
        """Post-rpc interceptor for update_snapshot

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_update_subscription(self, request: pubsub.UpdateSubscriptionRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[pubsub.UpdateSubscriptionRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_subscription

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_update_subscription(self, response: pubsub.Subscription) -> pubsub.Subscription:
        """Post-rpc interceptor for update_subscription

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_get_iam_policy(self, request: iam_policy_pb2.GetIamPolicyRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[iam_policy_pb2.GetIamPolicyRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_get_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for get_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_set_iam_policy(self, request: iam_policy_pb2.SetIamPolicyRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[iam_policy_pb2.SetIamPolicyRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_set_iam_policy(self, response: policy_pb2.Policy) -> policy_pb2.Policy:
        """Post-rpc interceptor for set_iam_policy

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response

    def pre_test_iam_permissions(self, request: iam_policy_pb2.TestIamPermissionsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[iam_policy_pb2.TestIamPermissionsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the request or metadata
        before they are sent to the Subscriber server.
        """
        return (request, metadata)

    def post_test_iam_permissions(self, response: iam_policy_pb2.TestIamPermissionsResponse) -> iam_policy_pb2.TestIamPermissionsResponse:
        """Post-rpc interceptor for test_iam_permissions

        Override in a subclass to manipulate the response
        after it is returned by the Subscriber server but before
        it is returned to user code.
        """
        return response