import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _list_bucket(bucket_name, prefix='', accept_key=lambda k: True, **session_kwargs):
    session = boto3.session.Session(**session_kwargs)
    client = session.client('s3')
    ctoken = None
    while True:
        if ctoken:
            kwargs = dict(Bucket=bucket_name, Prefix=prefix, ContinuationToken=ctoken)
        else:
            kwargs = dict(Bucket=bucket_name, Prefix=prefix)
        response = client.list_objects_v2(**kwargs)
        try:
            content = response['Contents']
        except KeyError:
            pass
        else:
            for c in content:
                key = c['Key']
                if accept_key(key):
                    yield key
        ctoken = response.get('NextContinuationToken', None)
        if not ctoken:
            break