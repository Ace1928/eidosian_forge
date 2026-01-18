import io
import json
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def _load_credentials_from_info(filename, info, scopes, default_scopes, quota_project_id, request):
    from google.auth.credentials import CredentialsWithQuotaProject
    credential_type = info.get('type')
    if credential_type == _AUTHORIZED_USER_TYPE:
        credentials, project_id = _get_authorized_user_credentials(filename, info, scopes)
    elif credential_type == _SERVICE_ACCOUNT_TYPE:
        credentials, project_id = _get_service_account_credentials(filename, info, scopes, default_scopes)
    elif credential_type == _EXTERNAL_ACCOUNT_TYPE:
        credentials, project_id = _get_external_account_credentials(info, filename, scopes=scopes, default_scopes=default_scopes, request=request)
    elif credential_type == _EXTERNAL_ACCOUNT_AUTHORIZED_USER_TYPE:
        credentials, project_id = _get_external_account_authorized_user_credentials(filename, info, request)
    elif credential_type == _IMPERSONATED_SERVICE_ACCOUNT_TYPE:
        credentials, project_id = _get_impersonated_service_account_credentials(filename, info, scopes)
    elif credential_type == _GDCH_SERVICE_ACCOUNT_TYPE:
        credentials, project_id = _get_gdch_service_account_credentials(filename, info)
    else:
        raise exceptions.DefaultCredentialsError('The file {file} does not have a valid type. Type is {type}, expected one of {valid_types}.'.format(file=filename, type=credential_type, valid_types=_VALID_TYPES))
    if isinstance(credentials, CredentialsWithQuotaProject):
        credentials = _apply_quota_project_id(credentials, quota_project_id)
    return (credentials, project_id)