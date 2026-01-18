import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _initialize_boto3(rw, client, client_kwargs, bucket, key):
    """Created the required objects for accessing S3.  Ideally, they have
    been already created for us and we can just reuse them."""
    if client_kwargs is None:
        client_kwargs = {}
    if client is None:
        init_kwargs = client_kwargs.get('S3.Client', {})
        client = boto3.client('s3', **init_kwargs)
    assert client
    rw._client = _ClientWrapper(client, client_kwargs)
    rw._bucket = bucket
    rw._key = key