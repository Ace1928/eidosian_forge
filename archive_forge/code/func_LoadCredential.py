import logging
import os
import subprocess
from typing import Iterator, List, Optional
from google.oauth2 import credentials as google_oauth2
import bq_auth_flags
import bq_flags
import bq_utils
from utils import bq_error
def LoadCredential() -> google_oauth2.Credentials:
    """Loads credentials by calling gcloud commands."""
    logging.info('Loading auth credentials from gcloud')
    gcloud_path = _GetGcloudPath()
    access_token = _GetAccessTokenAndPrintOutput(gcloud_path)
    refresh_token = _GetRefreshTokenAndPrintOutput(gcloud_path)
    return google_oauth2.Credentials(token=access_token, refresh_token=refresh_token, quota_project_id=bq_utils.GetResolvedQuotaProjectID(bq_auth_flags.QUOTA_PROJECT_ID.value, bq_flags.PROJECT_ID.value))