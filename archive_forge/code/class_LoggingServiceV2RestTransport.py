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
from googlecloudsdk.generated_clients.gapic_clients.logging_v2.types import logging
from .base import LoggingServiceV2Transport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
class LoggingServiceV2RestTransport(LoggingServiceV2Transport):
    """REST backend transport for LoggingServiceV2.

    Service for storing and querying logs.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    NOTE: This REST transport functionality is currently in a beta
    state (preview). We welcome your feedback via an issue in this
    library's source repository. Thank you!
    """

    def __init__(self, *, host: str='logging.googleapis.com', credentials: Optional[ga_credentials.Credentials]=None, credentials_file: Optional[str]=None, scopes: Optional[Sequence[str]]=None, client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]]=None, quota_project_id: Optional[str]=None, client_info: gapic_v1.client_info.ClientInfo=DEFAULT_CLIENT_INFO, always_use_jwt_access: Optional[bool]=False, url_scheme: str='https', interceptor: Optional[LoggingServiceV2RestInterceptor]=None, api_audience: Optional[str]=None) -> None:
        """Instantiate the transport.

       NOTE: This REST transport functionality is currently in a beta
       state (preview). We welcome your feedback via a GitHub issue in
       this library's repository. Thank you!

        Args:
            host (Optional[str]):
                 The hostname to connect to.
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        maybe_url_match = re.match('^(?P<scheme>http(?:s)?://)?(?P<host>.*)$', host)
        if maybe_url_match is None:
            raise ValueError(f'Unexpected hostname structure: {host}')
        url_match_items = maybe_url_match.groupdict()
        host = f'{url_scheme}://{host}' if not url_match_items['scheme'] else host
        super().__init__(host=host, credentials=credentials, client_info=client_info, always_use_jwt_access=always_use_jwt_access, api_audience=api_audience)
        self._session = AuthorizedSession(self._credentials, default_host=self.DEFAULT_HOST)
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or LoggingServiceV2RestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _DeleteLog(LoggingServiceV2RestStub):

        def __hash__(self):
            return hash('DeleteLog')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging.DeleteLogRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()):
            """Call the delete log method over HTTP.

            Args:
                request (~.logging.DeleteLogRequest):
                    The request object. The parameters to DeleteLog.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """
            http_options: List[Dict[str, str]] = [{'method': 'delete', 'uri': '/v2/{log_name=projects/*/logs/*}'}, {'method': 'delete', 'uri': '/v2/{log_name=*/*/logs/*}'}, {'method': 'delete', 'uri': '/v2/{log_name=organizations/*/logs/*}'}, {'method': 'delete', 'uri': '/v2/{log_name=folders/*/logs/*}'}, {'method': 'delete', 'uri': '/v2/{log_name=billingAccounts/*/logs/*}'}]
            request, metadata = self._interceptor.pre_delete_log(request, metadata)
            pb_request = logging.DeleteLogRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _ListLogEntries(LoggingServiceV2RestStub):

        def __hash__(self):
            return hash('ListLogEntries')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging.ListLogEntriesRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging.ListLogEntriesResponse:
            """Call the list log entries method over HTTP.

            Args:
                request (~.logging.ListLogEntriesRequest):
                    The request object. The parameters to ``ListLogEntries``.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging.ListLogEntriesResponse:
                    Result returned from ``ListLogEntries``.
            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v2beta1/entries:list', 'body': '*'}, {'method': 'post', 'uri': '/v2/entries:list', 'body': '*'}]
            request, metadata = self._interceptor.pre_list_log_entries(request, metadata)
            pb_request = logging.ListLogEntriesRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=False)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging.ListLogEntriesResponse()
            pb_resp = logging.ListLogEntriesResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_log_entries(resp)
            return resp

    class _ListLogs(LoggingServiceV2RestStub):

        def __hash__(self):
            return hash('ListLogs')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging.ListLogsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging.ListLogsResponse:
            """Call the list logs method over HTTP.

            Args:
                request (~.logging.ListLogsRequest):
                    The request object. The parameters to ListLogs.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging.ListLogsResponse:
                    Result returned from ListLogs.
            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v2/{parent=*/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=projects/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=organizations/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=folders/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=billingAccounts/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=projects/*/locations/*/buckets/*/views/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=organizations/*/locations/*/buckets/*/views/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=folders/*/locations/*/buckets/*/views/*}/logs'}, {'method': 'get', 'uri': '/v2/{parent=billingAccounts/*/locations/*/buckets/*/views/*}/logs'}]
            request, metadata = self._interceptor.pre_list_logs(request, metadata)
            pb_request = logging.ListLogsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging.ListLogsResponse()
            pb_resp = logging.ListLogsResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_logs(resp)
            return resp

    class _ListMonitoredResourceDescriptors(LoggingServiceV2RestStub):

        def __hash__(self):
            return hash('ListMonitoredResourceDescriptors')

        def __call__(self, request: logging.ListMonitoredResourceDescriptorsRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging.ListMonitoredResourceDescriptorsResponse:
            """Call the list monitored resource
        descriptors method over HTTP.

            Args:
                request (~.logging.ListMonitoredResourceDescriptorsRequest):
                    The request object. The parameters to
                ListMonitoredResourceDescriptors
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging.ListMonitoredResourceDescriptorsResponse:
                    Result returned from
                ListMonitoredResourceDescriptors.

            """
            http_options: List[Dict[str, str]] = [{'method': 'get', 'uri': '/v2/monitoredResourceDescriptors'}]
            request, metadata = self._interceptor.pre_list_monitored_resource_descriptors(request, metadata)
            pb_request = logging.ListMonitoredResourceDescriptorsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True))
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging.ListMonitoredResourceDescriptorsResponse()
            pb_resp = logging.ListMonitoredResourceDescriptorsResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_monitored_resource_descriptors(resp)
            return resp

    class _TailLogEntries(LoggingServiceV2RestStub):

        def __hash__(self):
            return hash('TailLogEntries')

        def __call__(self, request: logging.TailLogEntriesRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> rest_streaming.ResponseIterator:
            raise NotImplementedError('Method TailLogEntries is not available over REST transport')

    class _WriteLogEntries(LoggingServiceV2RestStub):

        def __hash__(self):
            return hash('WriteLogEntries')
        __REQUIRED_FIELDS_DEFAULT_VALUES: Dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {k: v for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items() if k not in message_dict}

        def __call__(self, request: logging.WriteLogEntriesRequest, *, retry: OptionalRetry=gapic_v1.method.DEFAULT, timeout: Optional[float]=None, metadata: Sequence[Tuple[str, str]]=()) -> logging.WriteLogEntriesResponse:
            """Call the write log entries method over HTTP.

            Args:
                request (~.logging.WriteLogEntriesRequest):
                    The request object. The parameters to WriteLogEntries.
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.logging.WriteLogEntriesResponse:
                    Result returned from WriteLogEntries.
            """
            http_options: List[Dict[str, str]] = [{'method': 'post', 'uri': '/v2/entries:write', 'body': '*'}, {'method': 'post', 'uri': '/v2beta1/entries:write', 'body': '*'}]
            request, metadata = self._interceptor.pre_write_log_entries(request, metadata)
            pb_request = logging.WriteLogEntriesRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            body = json_format.MessageToJson(transcoded_request['body'], including_default_value_fields=False, use_integers_for_enums=False)
            uri = transcoded_request['uri']
            method = transcoded_request['method']
            query_params = json.loads(json_format.MessageToJson(transcoded_request['query_params'], including_default_value_fields=False, use_integers_for_enums=False))
            query_params.update(self._get_unset_required_fields(query_params))
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)('{host}{uri}'.format(host=self._host, uri=uri), timeout=timeout, headers=headers, params=rest_helpers.flatten_query_params(query_params, strict=True), data=body)
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)
            resp = logging.WriteLogEntriesResponse()
            pb_resp = logging.WriteLogEntriesResponse.pb(resp)
            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_write_log_entries(resp)
            return resp

    @property
    def delete_log(self) -> Callable[[logging.DeleteLogRequest], empty_pb2.Empty]:
        return self._DeleteLog(self._session, self._host, self._interceptor)

    @property
    def list_log_entries(self) -> Callable[[logging.ListLogEntriesRequest], logging.ListLogEntriesResponse]:
        return self._ListLogEntries(self._session, self._host, self._interceptor)

    @property
    def list_logs(self) -> Callable[[logging.ListLogsRequest], logging.ListLogsResponse]:
        return self._ListLogs(self._session, self._host, self._interceptor)

    @property
    def list_monitored_resource_descriptors(self) -> Callable[[logging.ListMonitoredResourceDescriptorsRequest], logging.ListMonitoredResourceDescriptorsResponse]:
        return self._ListMonitoredResourceDescriptors(self._session, self._host, self._interceptor)

    @property
    def tail_log_entries(self) -> Callable[[logging.TailLogEntriesRequest], logging.TailLogEntriesResponse]:
        return self._TailLogEntries(self._session, self._host, self._interceptor)

    @property
    def write_log_entries(self) -> Callable[[logging.WriteLogEntriesRequest], logging.WriteLogEntriesResponse]:
        return self._WriteLogEntries(self._session, self._host, self._interceptor)

    @property
    def kind(self) -> str:
        return 'rest'

    def close(self):
        self._session.close()