import logging
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from tempfile import NamedTemporaryFile
from typing import IO, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse
from twisted.internet import defer, threads
from twisted.internet.defer import DeferredList
from w3lib.url import file_uri_to_path
from zope.interface import Interface, implementer
from scrapy import Spider, signals
from scrapy.exceptions import NotConfigured, ScrapyDeprecationWarning
from scrapy.extensions.postprocessing import PostProcessingManager
from scrapy.utils.boto import is_botocore_available
from scrapy.utils.conf import feed_complete_default_values_from_settings
from scrapy.utils.defer import maybe_deferred_to_future
from scrapy.utils.deprecate import create_deprecated_class
from scrapy.utils.ftp import ftp_store_file
from scrapy.utils.log import failure_to_exc_info
from scrapy.utils.misc import create_instance, load_object
from scrapy.utils.python import get_func_args, without_none_values
class S3FeedStorage(BlockingFeedStorage):

    def __init__(self, uri, access_key=None, secret_key=None, acl=None, endpoint_url=None, *, feed_options=None, session_token=None, region_name=None):
        if not is_botocore_available():
            raise NotConfigured('missing botocore library')
        u = urlparse(uri)
        self.bucketname = u.hostname
        self.access_key = u.username or access_key
        self.secret_key = u.password or secret_key
        self.session_token = session_token
        self.keyname = u.path[1:]
        self.acl = acl
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        if IS_BOTO3_AVAILABLE:
            import boto3.session
            session = boto3.session.Session()
            self.s3_client = session.client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key, aws_session_token=self.session_token, endpoint_url=self.endpoint_url, region_name=self.region_name)
        else:
            warnings.warn('`botocore` usage has been deprecated for S3 feed export, please use `boto3` to avoid problems', category=ScrapyDeprecationWarning)
            import botocore.session
            session = botocore.session.get_session()
            self.s3_client = session.create_client('s3', aws_access_key_id=self.access_key, aws_secret_access_key=self.secret_key, aws_session_token=self.session_token, endpoint_url=self.endpoint_url, region_name=self.region_name)
        if feed_options and feed_options.get('overwrite', True) is False:
            logger.warning('S3 does not support appending to files. To suppress this warning, remove the overwrite option from your FEEDS setting or set it to True.')

    @classmethod
    def from_crawler(cls, crawler, uri, *, feed_options=None):
        return build_storage(cls, uri, access_key=crawler.settings['AWS_ACCESS_KEY_ID'], secret_key=crawler.settings['AWS_SECRET_ACCESS_KEY'], session_token=crawler.settings['AWS_SESSION_TOKEN'], acl=crawler.settings['FEED_STORAGE_S3_ACL'] or None, endpoint_url=crawler.settings['AWS_ENDPOINT_URL'] or None, region_name=crawler.settings['AWS_REGION_NAME'] or None, feed_options=feed_options)

    def _store_in_thread(self, file):
        file.seek(0)
        if IS_BOTO3_AVAILABLE:
            kwargs = {'ExtraArgs': {'ACL': self.acl}} if self.acl else {}
            self.s3_client.upload_fileobj(Bucket=self.bucketname, Key=self.keyname, Fileobj=file, **kwargs)
        else:
            kwargs = {'ACL': self.acl} if self.acl else {}
            self.s3_client.put_object(Bucket=self.bucketname, Key=self.keyname, Body=file, **kwargs)
        file.close()