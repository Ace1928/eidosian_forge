import io
import json
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def _get_impersonated_service_account_credentials(filename, info, scopes):
    from google.auth import impersonated_credentials
    try:
        source_credentials_info = info.get('source_credentials')
        source_credentials_type = source_credentials_info.get('type')
        if source_credentials_type == _AUTHORIZED_USER_TYPE:
            source_credentials, _ = _get_authorized_user_credentials(filename, source_credentials_info)
        elif source_credentials_type == _SERVICE_ACCOUNT_TYPE:
            source_credentials, _ = _get_service_account_credentials(filename, source_credentials_info)
        else:
            raise exceptions.InvalidType('source credential of type {} is not supported.'.format(source_credentials_type))
        impersonation_url = info.get('service_account_impersonation_url')
        start_index = impersonation_url.rfind('/')
        end_index = impersonation_url.find(':generateAccessToken')
        if start_index == -1 or end_index == -1 or start_index > end_index:
            raise exceptions.InvalidValue('Cannot extract target principal from {}'.format(impersonation_url))
        target_principal = impersonation_url[start_index + 1:end_index]
        delegates = info.get('delegates')
        quota_project_id = info.get('quota_project_id')
        credentials = impersonated_credentials.Credentials(source_credentials, target_principal, scopes, delegates, quota_project_id=quota_project_id)
    except ValueError as caught_exc:
        msg = 'Failed to load impersonated service account credentials from {}'.format(filename)
        new_exc = exceptions.DefaultCredentialsError(msg, caught_exc)
        six.raise_from(new_exc, caught_exc)
    return (credentials, None)