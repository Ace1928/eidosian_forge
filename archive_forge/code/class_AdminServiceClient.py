from collections import OrderedDict
import os
import re
from typing import (
from google.cloud.pubsublite_v1 import gapic_version as package_version
from google.api_core import client_options as client_options_lib
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.api_core import retry as retries
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport import mtls  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth.exceptions import MutualTLSChannelError  # type: ignore
from google.oauth2 import service_account  # type: ignore
from google.api_core import operation  # type: ignore
from google.api_core import operation_async  # type: ignore
from google.cloud.pubsublite_v1.services.admin_service import pagers
from google.cloud.pubsublite_v1.types import admin
from google.cloud.pubsublite_v1.types import common
from google.longrunning import operations_pb2
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from .transports.base import AdminServiceTransport, DEFAULT_CLIENT_INFO
from .transports.grpc import AdminServiceGrpcTransport
from .transports.grpc_asyncio import AdminServiceGrpcAsyncIOTransport
class AdminServiceClient(metaclass=AdminServiceClientMeta):
    """The service that a client application uses to manage topics
    and subscriptions, such creating, listing, and deleting topics
    and subscriptions.
    """

    @staticmethod
    def _get_default_mtls_endpoint(api_endpoint):
        """Converts api endpoint to mTLS endpoint.

        Convert "*.sandbox.googleapis.com" and "*.googleapis.com" to
        "*.mtls.sandbox.googleapis.com" and "*.mtls.googleapis.com" respectively.
        Args:
            api_endpoint (Optional[str]): the api endpoint to convert.
        Returns:
            str: converted mTLS api endpoint.
        """
        if not api_endpoint:
            return api_endpoint
        mtls_endpoint_re = re.compile('(?P<name>[^.]+)(?P<mtls>\\.mtls)?(?P<sandbox>\\.sandbox)?(?P<googledomain>\\.googleapis\\.com)?')
        m = mtls_endpoint_re.match(api_endpoint)
        name, mtls, sandbox, googledomain = m.groups()
        if mtls or not googledomain:
            return api_endpoint
        if sandbox:
            return api_endpoint.replace('sandbox.googleapis.com', 'mtls.sandbox.googleapis.com')
        return api_endpoint.replace('.googleapis.com', '.mtls.googleapis.com')
    DEFAULT_ENDPOINT = 'pubsublite.googleapis.com'
    DEFAULT_MTLS_ENDPOINT = _get_default_mtls_endpoint.__func__(DEFAULT_ENDPOINT)

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            AdminServiceClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_info(info)
        kwargs['credentials'] = credentials
        return cls(*args, **kwargs)

    @classmethod
    def from_service_account_file(cls, filename: str, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            file.

        Args:
            filename (str): The path to the service account private key json
                file.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            AdminServiceClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_file(filename)
        kwargs['credentials'] = credentials
        return cls(*args, **kwargs)
    from_service_account_json = from_service_account_file

    @property
    def transport(self) -> AdminServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            AdminServiceTransport: The transport used by the client
                instance.
        """
        return self._transport

    @staticmethod
    def reservation_path(project: str, location: str, reservation: str) -> str:
        """Returns a fully-qualified reservation string."""
        return 'projects/{project}/locations/{location}/reservations/{reservation}'.format(project=project, location=location, reservation=reservation)

    @staticmethod
    def parse_reservation_path(path: str) -> Dict[str, str]:
        """Parses a reservation path into its component segments."""
        m = re.match('^projects/(?P<project>.+?)/locations/(?P<location>.+?)/reservations/(?P<reservation>.+?)$', path)
        return m.groupdict() if m else {}

    @staticmethod
    def subscription_path(project: str, location: str, subscription: str) -> str:
        """Returns a fully-qualified subscription string."""
        return 'projects/{project}/locations/{location}/subscriptions/{subscription}'.format(project=project, location=location, subscription=subscription)

    @staticmethod
    def parse_subscription_path(path: str) -> Dict[str, str]:
        """Parses a subscription path into its component segments."""
        m = re.match('^projects/(?P<project>.+?)/locations/(?P<location>.+?)/subscriptions/(?P<subscription>.+?)$', path)
        return m.groupdict() if m else {}

    @staticmethod
    def topic_path(project: str, location: str, topic: str) -> str:
        """Returns a fully-qualified topic string."""
        return 'projects/{project}/locations/{location}/topics/{topic}'.format(project=project, location=location, topic=topic)

    @staticmethod
    def parse_topic_path(path: str) -> Dict[str, str]:
        """Parses a topic path into its component segments."""
        m = re.match('^projects/(?P<project>.+?)/locations/(?P<location>.+?)/topics/(?P<topic>.+?)$', path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_billing_account_path(billing_account: str) -> str:
        """Returns a fully-qualified billing_account string."""
        return 'billingAccounts/{billing_account}'.format(billing_account=billing_account)

    @staticmethod
    def parse_common_billing_account_path(path: str) -> Dict[str, str]:
        """Parse a billing_account path into its component segments."""
        m = re.match('^billingAccounts/(?P<billing_account>.+?)$', path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_folder_path(folder: str) -> str:
        """Returns a fully-qualified folder string."""
        return 'folders/{folder}'.format(folder=folder)

    @staticmethod
    def parse_common_folder_path(path: str) -> Dict[str, str]:
        """Parse a folder path into its component segments."""
        m = re.match('^folders/(?P<folder>.+?)$', path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_organization_path(organization: str) -> str:
        """Returns a fully-qualified organization string."""
        return 'organizations/{organization}'.format(organization=organization)

    @staticmethod
    def parse_common_organization_path(path: str) -> Dict[str, str]:
        """Parse a organization path into its component segments."""
        m = re.match('^organizations/(?P<organization>.+?)$', path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_project_path(project: str) -> str:
        """Returns a fully-qualified project string."""
        return 'projects/{project}'.format(project=project)

    @staticmethod
    def parse_common_project_path(path: str) -> Dict[str, str]:
        """Parse a project path into its component segments."""
        m = re.match('^projects/(?P<project>.+?)$', path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_location_path(project: str, location: str) -> str:
        """Returns a fully-qualified location string."""
        return 'projects/{project}/locations/{location}'.format(project=project, location=location)

    @staticmethod
    def parse_common_location_path(path: str) -> Dict[str, str]:
        """Parse a location path into its component segments."""
        m = re.match('^projects/(?P<project>.+?)/locations/(?P<location>.+?)$', path)
        return m.groupdict() if m else {}

    @classmethod
    def get_mtls_endpoint_and_cert_source(cls, client_options: Optional[client_options_lib.ClientOptions]=None):
        """Return the API endpoint and client cert source for mutual TLS.

        The client cert source is determined in the following order:
        (1) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is not "true", the
        client cert source is None.
        (2) if `client_options.client_cert_source` is provided, use the provided one; if the
        default client cert source exists, use the default one; otherwise the client cert
        source is None.

        The API endpoint is determined in the following order:
        (1) if `client_options.api_endpoint` if provided, use the provided one.
        (2) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is "always", use the
        default mTLS endpoint; if the environment variable is "never", use the default API
        endpoint; otherwise if client cert source exists, use the default mTLS endpoint, otherwise
        use the default API endpoint.

        More details can be found at https://google.aip.dev/auth/4114.

        Args:
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. Only the `api_endpoint` and `client_cert_source` properties may be used
                in this method.

        Returns:
            Tuple[str, Callable[[], Tuple[bytes, bytes]]]: returns the API endpoint and the
                client cert source to use.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If any errors happen.
        """
        if client_options is None:
            client_options = client_options_lib.ClientOptions()
        use_client_cert = os.getenv('GOOGLE_API_USE_CLIENT_CERTIFICATE', 'false')
        use_mtls_endpoint = os.getenv('GOOGLE_API_USE_MTLS_ENDPOINT', 'auto')
        if use_client_cert not in ('true', 'false'):
            raise ValueError('Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or `false`')
        if use_mtls_endpoint not in ('auto', 'never', 'always'):
            raise MutualTLSChannelError('Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or `always`')
        client_cert_source = None
        if use_client_cert == 'true':
            if client_options.client_cert_source:
                client_cert_source = client_options.client_cert_source
            elif mtls.has_default_client_cert_source():
                client_cert_source = mtls.default_client_cert_source()
        if client_options.api_endpoint is not None:
            api_endpoint = client_options.api_endpoint
        elif use_mtls_endpoint == 'always' or (use_mtls_endpoint == 'auto' and client_cert_source):
            api_endpoint = cls.DEFAULT_MTLS_ENDPOINT
        else:
            api_endpoint = cls.DEFAULT_ENDPOINT
        return (api_endpoint, client_cert_source)

    def __init__(self, *, credentials: Optional[ga_credentials.Credentials]=None, transport: Optional[Union[str, AdminServiceTransport]]=None, client_options: Optional[Union[client_options_lib.ClientOptions, dict]]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO) -> None:
        """Instantiates the admin service client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Union[str, AdminServiceTransport]): The
                transport to use. If set to None, a transport is chosen
                automatically.
            client_options (Optional[Union[google.api_core.client_options.ClientOptions, dict]]): Custom options for the
                client. It won't take effect if a ``transport`` instance is provided.
                (1) The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client. GOOGLE_API_USE_MTLS_ENDPOINT
                environment variable can also be used to override the endpoint:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto switch to the
                default mTLS endpoint if client certificate is present, this is
                the default value). However, the ``api_endpoint`` property takes
                precedence if provided.
                (2) If GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide client certificate for mutual TLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        if isinstance(client_options, dict):
            client_options = client_options_lib.from_dict(client_options)
        if client_options is None:
            client_options = client_options_lib.ClientOptions()
        client_options = cast(client_options_lib.ClientOptions, client_options)
        api_endpoint, client_cert_source_func = self.get_mtls_endpoint_and_cert_source(client_options)
        api_key_value = getattr(client_options, 'api_key', None)
        if api_key_value and credentials:
            raise ValueError('client_options.api_key and credentials are mutually exclusive')
        if isinstance(transport, AdminServiceTransport):
            if credentials or client_options.credentials_file or api_key_value:
                raise ValueError('When providing a transport instance, provide its credentials directly.')
            if client_options.scopes:
                raise ValueError('When providing a transport instance, provide its scopes directly.')
            self._transport = transport
        else:
            import google.auth._default
            if api_key_value and hasattr(google.auth._default, 'get_api_key_credentials'):
                credentials = google.auth._default.get_api_key_credentials(api_key_value)
            Transport = type(self).get_transport_class(transport)
            self._transport = Transport(credentials=credentials, credentials_file=client_options.credentials_file, host=api_endpoint, scopes=client_options.scopes, client_cert_source_for_mtls=client_cert_source_func, quota_project_id=client_options.quota_project_id, client_info=client_info, always_use_jwt_access=True, api_audience=client_options.api_audience)

    def create_topic(self, request: Optional[Union[admin.CreateTopicRequest, dict]]=None, *, parent: Optional[str]=None, topic: Optional[common.Topic]=None, topic_id: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Topic:
        """Creates a new topic.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_create_topic():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.CreateTopicRequest(
                    parent="parent_value",
                    topic_id="topic_id_value",
                )

                # Make the request
                response = client.create_topic(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.CreateTopicRequest, dict]):
                The request object. Request for CreateTopic.
            parent (str):
                Required. The parent location in which to create the
                topic. Structured like
                ``projects/{project_number}/locations/{location}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            topic (google.cloud.pubsublite_v1.types.Topic):
                Required. Configuration of the topic to create. Its
                ``name`` field is ignored.

                This corresponds to the ``topic`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            topic_id (str):
                Required. The ID to use for the topic, which will become
                the final component of the topic's name.

                This value is structured like: ``my-topic-name``.

                This corresponds to the ``topic_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Topic:
                Metadata about a topic resource.
        """
        has_flattened_params = any([parent, topic, topic_id])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.CreateTopicRequest):
            request = admin.CreateTopicRequest(request)
            if parent is not None:
                request.parent = parent
            if topic is not None:
                request.topic = topic
            if topic_id is not None:
                request.topic_id = topic_id
        rpc = self._transport._wrapped_methods[self._transport.create_topic]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', request.parent),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def get_topic(self, request: Optional[Union[admin.GetTopicRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Topic:
        """Returns the topic configuration.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_get_topic():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.GetTopicRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_topic(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.GetTopicRequest, dict]):
                The request object. Request for GetTopic.
            name (str):
                Required. The name of the topic whose
                configuration to return.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Topic:
                Metadata about a topic resource.
        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.GetTopicRequest):
            request = admin.GetTopicRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.get_topic]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def get_topic_partitions(self, request: Optional[Union[admin.GetTopicPartitionsRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> admin.TopicPartitions:
        """Returns the partition information for the requested
        topic.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_get_topic_partitions():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.GetTopicPartitionsRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_topic_partitions(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.GetTopicPartitionsRequest, dict]):
                The request object. Request for GetTopicPartitions.
            name (str):
                Required. The topic whose partition
                information to return.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.TopicPartitions:
                Response for GetTopicPartitions.
        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.GetTopicPartitionsRequest):
            request = admin.GetTopicPartitionsRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.get_topic_partitions]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def list_topics(self, request: Optional[Union[admin.ListTopicsRequest, dict]]=None, *, parent: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> pagers.ListTopicsPager:
        """Returns the list of topics for the given project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_list_topics():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.ListTopicsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_topics(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.ListTopicsRequest, dict]):
                The request object. Request for ListTopics.
            parent (str):
                Required. The parent whose topics are to be listed.
                Structured like
                ``projects/{project_number}/locations/{location}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.services.admin_service.pagers.ListTopicsPager:
                Response for ListTopics.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        has_flattened_params = any([parent])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.ListTopicsRequest):
            request = admin.ListTopicsRequest(request)
            if parent is not None:
                request.parent = parent
        rpc = self._transport._wrapped_methods[self._transport.list_topics]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', request.parent),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        response = pagers.ListTopicsPager(method=rpc, request=request, response=response, metadata=metadata)
        return response

    def update_topic(self, request: Optional[Union[admin.UpdateTopicRequest, dict]]=None, *, topic: Optional[common.Topic]=None, update_mask: Optional[field_mask_pb2.FieldMask]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Topic:
        """Updates properties of the specified topic.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_update_topic():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.UpdateTopicRequest(
                )

                # Make the request
                response = client.update_topic(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.UpdateTopicRequest, dict]):
                The request object. Request for UpdateTopic.
            topic (google.cloud.pubsublite_v1.types.Topic):
                Required. The topic to update. Its ``name`` field must
                be populated.

                This corresponds to the ``topic`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (google.protobuf.field_mask_pb2.FieldMask):
                Required. A mask specifying the topic
                fields to change.

                This corresponds to the ``update_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Topic:
                Metadata about a topic resource.
        """
        has_flattened_params = any([topic, update_mask])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.UpdateTopicRequest):
            request = admin.UpdateTopicRequest(request)
            if topic is not None:
                request.topic = topic
            if update_mask is not None:
                request.update_mask = update_mask
        rpc = self._transport._wrapped_methods[self._transport.update_topic]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('topic.name', request.topic.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def delete_topic(self, request: Optional[Union[admin.DeleteTopicRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> None:
        """Deletes the specified topic.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_delete_topic():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.DeleteTopicRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_topic(request=request)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.DeleteTopicRequest, dict]):
                The request object. Request for DeleteTopic.
            name (str):
                Required. The name of the topic to
                delete.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.DeleteTopicRequest):
            request = admin.DeleteTopicRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.delete_topic]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def list_topic_subscriptions(self, request: Optional[Union[admin.ListTopicSubscriptionsRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> pagers.ListTopicSubscriptionsPager:
        """Lists the subscriptions attached to the specified
        topic.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_list_topic_subscriptions():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.ListTopicSubscriptionsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_topic_subscriptions(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.ListTopicSubscriptionsRequest, dict]):
                The request object. Request for ListTopicSubscriptions.
            name (str):
                Required. The name of the topic whose
                subscriptions to list.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.services.admin_service.pagers.ListTopicSubscriptionsPager:
                Response for ListTopicSubscriptions.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.ListTopicSubscriptionsRequest):
            request = admin.ListTopicSubscriptionsRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.list_topic_subscriptions]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        response = pagers.ListTopicSubscriptionsPager(method=rpc, request=request, response=response, metadata=metadata)
        return response

    def create_subscription(self, request: Optional[Union[admin.CreateSubscriptionRequest, dict]]=None, *, parent: Optional[str]=None, subscription: Optional[common.Subscription]=None, subscription_id: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Subscription:
        """Creates a new subscription.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_create_subscription():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.CreateSubscriptionRequest(
                    parent="parent_value",
                    subscription_id="subscription_id_value",
                )

                # Make the request
                response = client.create_subscription(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.CreateSubscriptionRequest, dict]):
                The request object. Request for CreateSubscription.
            parent (str):
                Required. The parent location in which to create the
                subscription. Structured like
                ``projects/{project_number}/locations/{location}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            subscription (google.cloud.pubsublite_v1.types.Subscription):
                Required. Configuration of the subscription to create.
                Its ``name`` field is ignored.

                This corresponds to the ``subscription`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            subscription_id (str):
                Required. The ID to use for the subscription, which will
                become the final component of the subscription's name.

                This value is structured like: ``my-sub-name``.

                This corresponds to the ``subscription_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Subscription:
                Metadata about a subscription
                resource.

        """
        has_flattened_params = any([parent, subscription, subscription_id])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.CreateSubscriptionRequest):
            request = admin.CreateSubscriptionRequest(request)
            if parent is not None:
                request.parent = parent
            if subscription is not None:
                request.subscription = subscription
            if subscription_id is not None:
                request.subscription_id = subscription_id
        rpc = self._transport._wrapped_methods[self._transport.create_subscription]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', request.parent),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def get_subscription(self, request: Optional[Union[admin.GetSubscriptionRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Subscription:
        """Returns the subscription configuration.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_get_subscription():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.GetSubscriptionRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_subscription(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.GetSubscriptionRequest, dict]):
                The request object. Request for GetSubscription.
            name (str):
                Required. The name of the
                subscription whose configuration to
                return.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Subscription:
                Metadata about a subscription
                resource.

        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.GetSubscriptionRequest):
            request = admin.GetSubscriptionRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.get_subscription]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def list_subscriptions(self, request: Optional[Union[admin.ListSubscriptionsRequest, dict]]=None, *, parent: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> pagers.ListSubscriptionsPager:
        """Returns the list of subscriptions for the given
        project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_list_subscriptions():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.ListSubscriptionsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_subscriptions(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.ListSubscriptionsRequest, dict]):
                The request object. Request for ListSubscriptions.
            parent (str):
                Required. The parent whose subscriptions are to be
                listed. Structured like
                ``projects/{project_number}/locations/{location}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.services.admin_service.pagers.ListSubscriptionsPager:
                Response for ListSubscriptions.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        has_flattened_params = any([parent])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.ListSubscriptionsRequest):
            request = admin.ListSubscriptionsRequest(request)
            if parent is not None:
                request.parent = parent
        rpc = self._transport._wrapped_methods[self._transport.list_subscriptions]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', request.parent),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        response = pagers.ListSubscriptionsPager(method=rpc, request=request, response=response, metadata=metadata)
        return response

    def update_subscription(self, request: Optional[Union[admin.UpdateSubscriptionRequest, dict]]=None, *, subscription: Optional[common.Subscription]=None, update_mask: Optional[field_mask_pb2.FieldMask]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Subscription:
        """Updates properties of the specified subscription.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_update_subscription():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.UpdateSubscriptionRequest(
                )

                # Make the request
                response = client.update_subscription(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.UpdateSubscriptionRequest, dict]):
                The request object. Request for UpdateSubscription.
            subscription (google.cloud.pubsublite_v1.types.Subscription):
                Required. The subscription to update. Its ``name`` field
                must be populated. Topic field must not be populated.

                This corresponds to the ``subscription`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (google.protobuf.field_mask_pb2.FieldMask):
                Required. A mask specifying the
                subscription fields to change.

                This corresponds to the ``update_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Subscription:
                Metadata about a subscription
                resource.

        """
        has_flattened_params = any([subscription, update_mask])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.UpdateSubscriptionRequest):
            request = admin.UpdateSubscriptionRequest(request)
            if subscription is not None:
                request.subscription = subscription
            if update_mask is not None:
                request.update_mask = update_mask
        rpc = self._transport._wrapped_methods[self._transport.update_subscription]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('subscription.name', request.subscription.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def delete_subscription(self, request: Optional[Union[admin.DeleteSubscriptionRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> None:
        """Deletes the specified subscription.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_delete_subscription():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.DeleteSubscriptionRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_subscription(request=request)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.DeleteSubscriptionRequest, dict]):
                The request object. Request for DeleteSubscription.
            name (str):
                Required. The name of the
                subscription to delete.

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.DeleteSubscriptionRequest):
            request = admin.DeleteSubscriptionRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.delete_subscription]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def seek_subscription(self, request: Optional[Union[admin.SeekSubscriptionRequest, dict]]=None, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> operation.Operation:
        """Performs an out-of-band seek for a subscription to a
        specified target, which may be timestamps or named
        positions within the message backlog. Seek translates
        these targets to cursors for each partition and
        orchestrates subscribers to start consuming messages
        from these seek cursors.

        If an operation is returned, the seek has been
        registered and subscribers will eventually receive
        messages from the seek cursors (i.e. eventual
        consistency), as long as they are using a minimum
        supported client library version and not a system that
        tracks cursors independently of Pub/Sub Lite (e.g.
        Apache Beam, Dataflow, Spark). The seek operation will
        fail for unsupported clients.

        If clients would like to know when subscribers react to
        the seek (or not), they can poll the operation. The seek
        operation will succeed and complete once subscribers are
        ready to receive messages from the seek cursors for all
        partitions of the topic. This means that the seek
        operation will not complete until all subscribers come
        online.

        If the previous seek operation has not yet completed, it
        will be aborted and the new invocation of seek will
        supersede it.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_seek_subscription():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.SeekSubscriptionRequest(
                    named_target="HEAD",
                    name="name_value",
                )

                # Make the request
                operation = client.seek_subscription(request=request)

                print("Waiting for operation to complete...")

                response = operation.result()

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.SeekSubscriptionRequest, dict]):
                The request object. Request for SeekSubscription.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.api_core.operation.Operation:
                An object representing a long-running operation.

                The result type for the operation will be
                :class:`google.cloud.pubsublite_v1.types.SeekSubscriptionResponse`
                Response for SeekSubscription long running operation.

        """
        if not isinstance(request, admin.SeekSubscriptionRequest):
            request = admin.SeekSubscriptionRequest(request)
        rpc = self._transport._wrapped_methods[self._transport.seek_subscription]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        response = operation.from_gapic(response, self._transport.operations_client, admin.SeekSubscriptionResponse, metadata_type=admin.OperationMetadata)
        return response

    def create_reservation(self, request: Optional[Union[admin.CreateReservationRequest, dict]]=None, *, parent: Optional[str]=None, reservation: Optional[common.Reservation]=None, reservation_id: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Reservation:
        """Creates a new reservation.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_create_reservation():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.CreateReservationRequest(
                    parent="parent_value",
                    reservation_id="reservation_id_value",
                )

                # Make the request
                response = client.create_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.CreateReservationRequest, dict]):
                The request object. Request for CreateReservation.
            parent (str):
                Required. The parent location in which to create the
                reservation. Structured like
                ``projects/{project_number}/locations/{location}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            reservation (google.cloud.pubsublite_v1.types.Reservation):
                Required. Configuration of the reservation to create.
                Its ``name`` field is ignored.

                This corresponds to the ``reservation`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            reservation_id (str):
                Required. The ID to use for the reservation, which will
                become the final component of the reservation's name.

                This value is structured like: ``my-reservation-name``.

                This corresponds to the ``reservation_id`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Reservation:
                Metadata about a reservation
                resource.

        """
        has_flattened_params = any([parent, reservation, reservation_id])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.CreateReservationRequest):
            request = admin.CreateReservationRequest(request)
            if parent is not None:
                request.parent = parent
            if reservation is not None:
                request.reservation = reservation
            if reservation_id is not None:
                request.reservation_id = reservation_id
        rpc = self._transport._wrapped_methods[self._transport.create_reservation]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', request.parent),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def get_reservation(self, request: Optional[Union[admin.GetReservationRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Reservation:
        """Returns the reservation configuration.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_get_reservation():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.GetReservationRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.GetReservationRequest, dict]):
                The request object. Request for GetReservation.
            name (str):
                Required. The name of the reservation whose
                configuration to return. Structured like:
                projects/{project_number}/locations/{location}/reservations/{reservation_id}

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Reservation:
                Metadata about a reservation
                resource.

        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.GetReservationRequest):
            request = admin.GetReservationRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.get_reservation]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def list_reservations(self, request: Optional[Union[admin.ListReservationsRequest, dict]]=None, *, parent: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> pagers.ListReservationsPager:
        """Returns the list of reservations for the given
        project.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_list_reservations():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.ListReservationsRequest(
                    parent="parent_value",
                )

                # Make the request
                page_result = client.list_reservations(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.ListReservationsRequest, dict]):
                The request object. Request for ListReservations.
            parent (str):
                Required. The parent whose reservations are to be
                listed. Structured like
                ``projects/{project_number}/locations/{location}``.

                This corresponds to the ``parent`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.services.admin_service.pagers.ListReservationsPager:
                Response for ListReservations.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        has_flattened_params = any([parent])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.ListReservationsRequest):
            request = admin.ListReservationsRequest(request)
            if parent is not None:
                request.parent = parent
        rpc = self._transport._wrapped_methods[self._transport.list_reservations]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', request.parent),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        response = pagers.ListReservationsPager(method=rpc, request=request, response=response, metadata=metadata)
        return response

    def update_reservation(self, request: Optional[Union[admin.UpdateReservationRequest, dict]]=None, *, reservation: Optional[common.Reservation]=None, update_mask: Optional[field_mask_pb2.FieldMask]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> common.Reservation:
        """Updates properties of the specified reservation.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_update_reservation():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.UpdateReservationRequest(
                )

                # Make the request
                response = client.update_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.UpdateReservationRequest, dict]):
                The request object. Request for UpdateReservation.
            reservation (google.cloud.pubsublite_v1.types.Reservation):
                Required. The reservation to update. Its ``name`` field
                must be populated.

                This corresponds to the ``reservation`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            update_mask (google.protobuf.field_mask_pb2.FieldMask):
                Required. A mask specifying the
                reservation fields to change.

                This corresponds to the ``update_mask`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.types.Reservation:
                Metadata about a reservation
                resource.

        """
        has_flattened_params = any([reservation, update_mask])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.UpdateReservationRequest):
            request = admin.UpdateReservationRequest(request)
            if reservation is not None:
                request.reservation = reservation
            if update_mask is not None:
                request.update_mask = update_mask
        rpc = self._transport._wrapped_methods[self._transport.update_reservation]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('reservation.name', request.reservation.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def delete_reservation(self, request: Optional[Union[admin.DeleteReservationRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> None:
        """Deletes the specified reservation.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_delete_reservation():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.DeleteReservationRequest(
                    name="name_value",
                )

                # Make the request
                client.delete_reservation(request=request)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.DeleteReservationRequest, dict]):
                The request object. Request for DeleteReservation.
            name (str):
                Required. The name of the reservation to delete.
                Structured like:
                projects/{project_number}/locations/{location}/reservations/{reservation_id}

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.DeleteReservationRequest):
            request = admin.DeleteReservationRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.delete_reservation]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def list_reservation_topics(self, request: Optional[Union[admin.ListReservationTopicsRequest, dict]]=None, *, name: Optional[str]=None, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> pagers.ListReservationTopicsPager:
        """Lists the topics attached to the specified
        reservation.

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import pubsublite_v1

            def sample_list_reservation_topics():
                # Create a client
                client = pubsublite_v1.AdminServiceClient()

                # Initialize request argument(s)
                request = pubsublite_v1.ListReservationTopicsRequest(
                    name="name_value",
                )

                # Make the request
                page_result = client.list_reservation_topics(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.pubsublite_v1.types.ListReservationTopicsRequest, dict]):
                The request object. Request for ListReservationTopics.
            name (str):
                Required. The name of the reservation whose topics to
                list. Structured like:
                projects/{project_number}/locations/{location}/reservations/{reservation_id}

                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.pubsublite_v1.services.admin_service.pagers.ListReservationTopicsPager:
                Response for ListReservationTopics.
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        has_flattened_params = any([name])
        if request is not None and has_flattened_params:
            raise ValueError('If the `request` argument is set, then none of the individual field arguments should be set.')
        if not isinstance(request, admin.ListReservationTopicsRequest):
            request = admin.ListReservationTopicsRequest(request)
            if name is not None:
                request.name = name
        rpc = self._transport._wrapped_methods[self._transport.list_reservation_topics]
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        response = pagers.ListReservationTopicsPager(method=rpc, request=request, response=response, metadata=metadata)
        return response

    def __enter__(self) -> 'AdminServiceClient':
        return self

    def __exit__(self, type, value, traceback):
        """Releases underlying transport's resources.

        .. warning::
            ONLY use as a context manager if the transport is NOT shared
            with other clients! Exiting the with block will CLOSE the transport
            and may cause errors in other clients!
        """
        self.transport.close()

    def list_operations(self, request: Optional[operations_pb2.ListOperationsRequest]=None, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> operations_pb2.ListOperationsResponse:
        """Lists operations that match the specified filter in the request.

        Args:
            request (:class:`~.operations_pb2.ListOperationsRequest`):
                The request object. Request message for
                `ListOperations` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        Returns:
            ~.operations_pb2.ListOperationsResponse:
                Response message for ``ListOperations`` method.
        """
        if isinstance(request, dict):
            request = operations_pb2.ListOperationsRequest(**request)
        rpc = gapic_v1.method.wrap_method(self._transport.list_operations, default_timeout=None, client_info=DEFAULT_CLIENT_INFO)
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def get_operation(self, request: Optional[operations_pb2.GetOperationRequest]=None, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> operations_pb2.Operation:
        """Gets the latest state of a long-running operation.

        Args:
            request (:class:`~.operations_pb2.GetOperationRequest`):
                The request object. Request message for
                `GetOperation` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        Returns:
            ~.operations_pb2.Operation:
                An ``Operation`` object.
        """
        if isinstance(request, dict):
            request = operations_pb2.GetOperationRequest(**request)
        rpc = gapic_v1.method.wrap_method(self._transport.get_operation, default_timeout=None, client_info=DEFAULT_CLIENT_INFO)
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)
        return response

    def delete_operation(self, request: Optional[operations_pb2.DeleteOperationRequest]=None, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> None:
        """Deletes a long-running operation.

        This method indicates that the client is no longer interested
        in the operation result. It does not cancel the operation.
        If the server doesn't support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.

        Args:
            request (:class:`~.operations_pb2.DeleteOperationRequest`):
                The request object. Request message for
                `DeleteOperation` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        Returns:
            None
        """
        if isinstance(request, dict):
            request = operations_pb2.DeleteOperationRequest(**request)
        rpc = gapic_v1.method.wrap_method(self._transport.delete_operation, default_timeout=None, client_info=DEFAULT_CLIENT_INFO)
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def cancel_operation(self, request: Optional[operations_pb2.CancelOperationRequest]=None, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Union[float, object]=gapic_v1.method.DEFAULT, metadata: Sequence[Tuple[str, str]]=()) -> None:
        """Starts asynchronous cancellation on a long-running operation.

        The server makes a best effort to cancel the operation, but success
        is not guaranteed.  If the server doesn't support this method, it returns
        `google.rpc.Code.UNIMPLEMENTED`.

        Args:
            request (:class:`~.operations_pb2.CancelOperationRequest`):
                The request object. Request message for
                `CancelOperation` method.
            retry (google.api_core.retry.Retry): Designation of what errors,
                    if any, should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        Returns:
            None
        """
        if isinstance(request, dict):
            request = operations_pb2.CancelOperationRequest(**request)
        rpc = gapic_v1.method.wrap_method(self._transport.cancel_operation, default_timeout=None, client_info=DEFAULT_CLIENT_INFO)
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', request.name),)),)
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)