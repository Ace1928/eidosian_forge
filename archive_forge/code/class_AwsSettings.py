import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
class AwsSettings(BaseSettings):
    aws_access_token: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = 'us-east-1'
    set_s3_endpoint: Optional[bool] = True
    s3_config: Optional[Union[str, Dict[str, Any]]] = None
    s3_bucket: Optional[str] = None
    s3_backup_bucket: Optional[str] = None

    @validator('s3_config', pre=True)
    def validate_s3_config(cls, v):
        if v is None:
            return {}
        return json.loads(v) if isinstance(v, str) else v

    @lazyproperty
    def s3_endpoint(self):
        return f'https://s3.{self.aws_region}.amazonaws.com'

    @lazyproperty
    @require_fileio()
    def s3_bucket_path(self):
        if self.s3_bucket is None:
            return None
        bucket = self.s3_bucket
        if not bucket.startswith('s3://'):
            bucket = f's3://{bucket}'
        return File(bucket)

    @lazyproperty
    @require_fileio()
    def s3_backup_bucket_path(self):
        if self.s3_backup_bucket is None:
            return None
        bucket = self.s3_backup_bucket
        if not bucket.startswith('s3://'):
            bucket = f's3://{bucket}'
        return File(bucket)

    def set_env(self):
        if self.aws_access_key_id:
            os.environ['AWS_ACCESS_KEY_ID'] = self.aws_access_key_id
        if self.aws_secret_access_key:
            os.environ['AWS_SECRET_ACCESS_KEY'] = self.aws_secret_access_key
        if self.aws_region:
            os.environ['AWS_REGION'] = self.aws_region
        if self.aws_access_token:
            os.environ['AWS_ACCESS_TOKEN'] = self.aws_access_token
        if self.set_s3_endpoint:
            os.environ['S3_ENDPOINT'] = self.s3_endpoint