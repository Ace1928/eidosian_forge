import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
class MinioSettings(BaseSettings):
    minio_endpoint: Optional[str] = None
    minio_access_key: Optional[str] = None
    minio_secret_key: Optional[str] = None
    minio_access_token: Optional[str] = None
    minio_secure: Optional[bool] = True
    minio_region: Optional[str] = None
    minio_config: Optional[Union[str, Dict[str, Any]]] = None
    minio_signature_ver: Optional[str] = 's3v4'
    minio_bucket: Optional[str] = None
    minio_backup_bucket: Optional[str] = None

    @validator('minio_config', pre=True)
    def validate_minio_config(cls, v):
        if v is None:
            return {}
        return json.loads(v) if isinstance(v, str) else v

    @lazyproperty
    @require_fileio()
    def minio_bucket_path(self):
        if self.minio_bucket is None:
            return None
        bucket = self.minio_bucket
        if not bucket.startswith('minio://'):
            bucket = f'minio://{bucket}'
        return File(bucket)

    @lazyproperty
    @require_fileio()
    def minio_backup_bucket_path(self):
        if self.minio_backup_bucket is None:
            return None
        bucket = self.minio_backup_bucket
        if not bucket.startswith('minio://'):
            bucket = f'minio://{bucket}'
        return File(bucket)

    def set_env(self):
        if self.minio_endpoint:
            os.environ['MINIO_ENDPOINT'] = self.minio_endpoint
        if self.minio_access_key:
            os.environ['MINIO_ACCESS_KEY'] = self.minio_access_key
        if self.minio_secret_key:
            os.environ['MINIO_SECRET_KEY'] = self.minio_secret_key
        if self.minio_secure:
            os.environ['MINIO_SECURE'] = str(self.minio_secure)
        if self.minio_region:
            os.environ['MINIO_REGION'] = self.minio_region
        if self.minio_signature_ver:
            os.environ['MINIO_SIGNATURE_VER'] = self.minio_signature_ver