import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _download_fileobj(bucket, key_name):
    buf = io.BytesIO()
    bucket.download_fileobj(key_name, buf)
    return buf.getvalue()