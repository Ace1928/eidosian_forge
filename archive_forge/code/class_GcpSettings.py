import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
class GcpSettings(BaseSettings):
    gcp_project: Optional[str] = None
    gcloud_project: Optional[str] = None
    google_cloud_project: Optional[str] = None
    google_application_credentials: Optional[Union[str, pathlib.Path]] = None
    gcs_client_config: Optional[Union[str, Dict[str, Any]]] = None
    gcs_config: Optional[Union[str, Dict[str, Any]]] = None
    gs_bucket: Optional[str] = None
    gs_backup_bucket: Optional[str] = None

    @validator('google_application_credentials')
    def validate_google_application_credentials(cls, v):
        if v is None:
            return pathlib.Path.home().joinpath('adc.json')
        if _fileio_available:
            return File(v)
        if isinstance(v, str):
            v = pathlib.Path(v)
        return v

    @validator('gcs_client_config')
    def validate_gcs_client_config(cls, v) -> Dict:
        if v is None:
            return {}
        return json.loads(v) if isinstance(v, str) else v

    @validator('gcs_config')
    def validate_gcs_config(cls, v) -> Dict:
        if v is None:
            return {}
        return json.loads(v) if isinstance(v, str) else v

    @lazyproperty
    def adc_exists(self):
        return self.google_application_credentials.exists()

    @lazyproperty
    def project(self):
        return self.gcp_project or self.gcloud_project or self.google_cloud_project

    @lazyproperty
    @require_fileio()
    def gs_bucket_path(self):
        if self.gs_bucket is None:
            return None
        bucket = self.gs_bucket
        if not bucket.startswith('gs://'):
            bucket = f'gs://{bucket}'
        return File(bucket)

    @lazyproperty
    @require_fileio()
    def gs_backup_bucket_path(self):
        if self.gs_backup_bucket is None:
            return None
        bucket = self.gs_backup_bucket
        if not bucket.startswith('gs://'):
            bucket = f'gs://{bucket}'
        return File(bucket)

    def set_env(self):
        if self.adc_exists:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.google_application_credentials.as_posix()
        if self.project:
            os.environ['GOOGLE_CLOUD_PROJECT'] = self.project