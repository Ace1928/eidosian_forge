import base64
import functools
import hashlib
import logging
import mimetypes
import os
import time
from collections import defaultdict
from contextlib import suppress
from ftplib import FTP
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import DefaultDict, Optional, Set, Union
from urllib.parse import urlparse
from itemadapter import ItemAdapter
from twisted.internet import defer, threads
from scrapy.exceptions import IgnoreRequest, NotConfigured
from scrapy.http import Request
from scrapy.http.request import NO_CALLBACK
from scrapy.pipelines.media import MediaPipeline
from scrapy.settings import Settings
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.datatypes import CaseInsensitiveDict
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import md5sum
from scrapy.utils.python import to_bytes
from scrapy.utils.request import referer_str
class GCSFilesStore:
    GCS_PROJECT_ID = None
    CACHE_CONTROL = 'max-age=172800'
    POLICY = None

    def __init__(self, uri):
        from google.cloud import storage
        client = storage.Client(project=self.GCS_PROJECT_ID)
        bucket, prefix = uri[5:].split('/', 1)
        self.bucket = client.bucket(bucket)
        self.prefix = prefix
        permissions = self.bucket.test_iam_permissions(['storage.objects.get', 'storage.objects.create'])
        if 'storage.objects.get' not in permissions:
            logger.warning("No 'storage.objects.get' permission for GSC bucket %(bucket)s. Checking if files are up to date will be impossible. Files will be downloaded every time.", {'bucket': bucket})
        if 'storage.objects.create' not in permissions:
            logger.error("No 'storage.objects.create' permission for GSC bucket %(bucket)s. Saving files will be impossible!", {'bucket': bucket})

    def stat_file(self, path, info):

        def _onsuccess(blob):
            if blob:
                checksum = base64.b64decode(blob.md5_hash).hex()
                last_modified = time.mktime(blob.updated.timetuple())
                return {'checksum': checksum, 'last_modified': last_modified}
            return {}
        blob_path = self._get_blob_path(path)
        return threads.deferToThread(self.bucket.get_blob, blob_path).addCallback(_onsuccess)

    def _get_content_type(self, headers):
        if headers and 'Content-Type' in headers:
            return headers['Content-Type']
        return 'application/octet-stream'

    def _get_blob_path(self, path):
        return self.prefix + path

    def persist_file(self, path, buf, info, meta=None, headers=None):
        blob_path = self._get_blob_path(path)
        blob = self.bucket.blob(blob_path)
        blob.cache_control = self.CACHE_CONTROL
        blob.metadata = {k: str(v) for k, v in (meta or {}).items()}
        return threads.deferToThread(blob.upload_from_string, data=buf.getvalue(), content_type=self._get_content_type(headers), predefined_acl=self.POLICY)