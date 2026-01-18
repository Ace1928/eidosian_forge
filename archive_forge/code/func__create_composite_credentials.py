from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
def _create_composite_credentials(credentials=None, credentials_file=None, default_scopes=None, scopes=None, ssl_credentials=None, quota_project_id=None, default_host=None):
    """Create the composite credentials for secure channels.

    Args:
        credentials (google.auth.credentials.Credentials): The credentials. If
            not specified, then this function will attempt to ascertain the
            credentials from the environment using :func:`google.auth.default`.
        credentials_file (str): A file with credentials that can be loaded with
            :func:`google.auth.load_credentials_from_file`. This argument is
            mutually exclusive with credentials.
        default_scopes (Sequence[str]): A optional list of scopes needed for this
            service. These are only used when credentials are not specified and
            are passed to :func:`google.auth.default`.
        scopes (Sequence[str]): A optional list of scopes needed for this
            service. These are only used when credentials are not specified and
            are passed to :func:`google.auth.default`.
        ssl_credentials (grpc.ChannelCredentials): Optional SSL channel
            credentials. This can be used to specify different certificates.
        quota_project_id (str): An optional project to use for billing and quota.
        default_host (str): The default endpoint. e.g., "pubsub.googleapis.com".

    Returns:
        grpc.ChannelCredentials: The composed channel credentials object.

    Raises:
        google.api_core.DuplicateCredentialArgs: If both a credentials object and credentials_file are passed.
    """
    if credentials and credentials_file:
        raise exceptions.DuplicateCredentialArgs("'credentials' and 'credentials_file' are mutually exclusive.")
    if credentials_file:
        credentials, _ = google.auth.load_credentials_from_file(credentials_file, scopes=scopes, default_scopes=default_scopes)
    elif credentials:
        credentials = google.auth.credentials.with_scopes_if_required(credentials, scopes=scopes, default_scopes=default_scopes)
    else:
        credentials, _ = google.auth.default(scopes=scopes, default_scopes=default_scopes)
    if quota_project_id and isinstance(credentials, google.auth.credentials.CredentialsWithQuotaProject):
        credentials = credentials.with_quota_project(quota_project_id)
    request = google.auth.transport.requests.Request()
    metadata_plugin = google.auth.transport.grpc.AuthMetadataPlugin(credentials, request, default_host=default_host)
    google_auth_credentials = grpc.metadata_call_credentials(metadata_plugin)
    if ssl_credentials:
        return grpc.composite_channel_credentials(ssl_credentials, google_auth_credentials)
    else:
        return grpc.compute_engine_channel_credentials(google_auth_credentials)