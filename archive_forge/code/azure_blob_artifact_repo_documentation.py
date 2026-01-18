import base64
import datetime
import os
import posixpath
import re
import urllib.parse
from typing import Union
from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
from mlflow.environment_variables import MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository, MultipartUploadMixin
from mlflow.utils.credentials import get_default_host_creds
Parse a wasbs:// URI, returning (container, storage_account, path, api_uri_suffix).