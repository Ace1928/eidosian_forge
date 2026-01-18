import io
import json
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.transport._http_client
def _apply_quota_project_id(credentials, quota_project_id):
    if quota_project_id:
        credentials = credentials.with_quota_project(quota_project_id)
    else:
        credentials = credentials.with_quota_project_from_environment()
    from google.oauth2 import credentials as authorized_user_credentials
    if isinstance(credentials, authorized_user_credentials.Credentials) and (not credentials.quota_project_id):
        _warn_about_problematic_credentials(credentials)
    return credentials