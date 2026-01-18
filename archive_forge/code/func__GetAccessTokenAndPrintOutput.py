import logging
import os
import subprocess
from typing import Iterator, List, Optional
from google.oauth2 import credentials as google_oauth2
import bq_auth_flags
import bq_flags
import bq_utils
from utils import bq_error
def _GetAccessTokenAndPrintOutput(gcloud_path: str) -> Optional[str]:
    return _GetTokenFromGcloudAndPrintOtherOutput([gcloud_path, 'auth', 'print-access-token'])