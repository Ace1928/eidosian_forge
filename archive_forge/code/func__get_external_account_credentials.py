import io
import json
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def _get_external_account_credentials(info, filename, scopes=None, default_scopes=None, request=None):
    """Loads external account Credentials from the parsed external account info.

    The credentials information must correspond to a supported external account
    credentials.

    Args:
        info (Mapping[str, str]): The external account info in Google format.
        filename (str): The full path to the credentials file.
        scopes (Optional[Sequence[str]]): The list of scopes for the credentials. If
            specified, the credentials will automatically be scoped if
            necessary.
        default_scopes (Optional[Sequence[str]]): Default scopes passed by a
            Google client library. Use 'scopes' for user-defined scopes.
        request (Optional[google.auth.transport.Request]): An object used to make
            HTTP requests. This is used to determine the associated project ID
            for a workload identity pool resource (external account credentials).
            If not specified, then it will use a
            google.auth.transport.requests.Request client to make requests.

    Returns:
        Tuple[google.auth.credentials.Credentials, Optional[str]]: Loaded
            credentials and the project ID. External account credentials project
            IDs may not always be determined.

    Raises:
        google.auth.exceptions.DefaultCredentialsError: if the info dictionary
            is in the wrong format or is missing required information.
    """
    if info.get('subject_token_type') == _AWS_SUBJECT_TOKEN_TYPE:
        from google.auth import aws
        credentials = aws.Credentials.from_info(info, scopes=scopes, default_scopes=default_scopes)
    elif info.get('credential_source') is not None and info.get('credential_source').get('executable') is not None:
        from google.auth import pluggable
        credentials = pluggable.Credentials.from_info(info, scopes=scopes, default_scopes=default_scopes)
    else:
        try:
            from google.auth import identity_pool
            credentials = identity_pool.Credentials.from_info(info, scopes=scopes, default_scopes=default_scopes)
        except ValueError:
            raise exceptions.DefaultCredentialsError('Failed to load external account credentials from {}'.format(filename))
    if request is None:
        import google.auth.transport.requests
        request = google.auth.transport.requests.Request()
    return (credentials, credentials.get_project_id(request=request))