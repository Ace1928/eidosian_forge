import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _download_key(key_name, bucket_name=None, retries=3, **session_kwargs):
    if bucket_name is None:
        raise ValueError('bucket_name may not be None')
    session = boto3.session.Session(**session_kwargs)
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for x in range(retries + 1):
        try:
            content_bytes = _download_fileobj(bucket, key_name)
        except botocore.client.ClientError:
            if x == retries:
                raise
            pass
        else:
            return (key_name, content_bytes)