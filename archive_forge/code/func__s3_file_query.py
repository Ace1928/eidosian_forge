import mimetypes
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from ..client import Client, register_client_class
from ..cloudpath import implementation_registry
from ..enums import FileCacheMode
from ..exceptions import CloudPathException
from .s3path import S3Path
def _s3_file_query(self, cloud_path: S3Path):
    """Boto3 query used for quick checks of existence and if path is file/dir"""
    try:
        self.client.head_object(Bucket=cloud_path.bucket, Key=cloud_path.key.rstrip('/'), **self.boto3_dl_extra_args)
        return 'file'
    except (ClientError, self.client.exceptions.NoSuchKey):
        key = cloud_path.key.rstrip('/') + '/'
        return next(('dir' for obj in self.s3.Bucket(cloud_path.bucket).objects.filter(Prefix=key, **self.boto3_list_extra_args).limit(1)), None)