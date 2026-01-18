import subprocess
import time
import logging.handlers
import boto
import boto.provider
import collections
import tempfile
import random
import smtplib
import datetime
import re
import io
import email.mime.multipart
import email.mime.base
import email.mime.text
import email.utils
import email.encoders
import gzip
import threading
import locale
import sys
from boto.compat import six, StringIO, urllib, encodebytes
from contextlib import contextmanager
from hashlib import md5, sha512
from boto.compat import json
class LazyLoadMetadata(dict):

    def __init__(self, url, num_retries, timeout=None):
        self._url = url
        self._num_retries = num_retries
        self._leaves = {}
        self._dicts = []
        self._timeout = timeout
        data = boto.utils.retry_url(self._url, num_retries=self._num_retries, timeout=self._timeout)
        if data:
            fields = data.split('\n')
            for field in fields:
                if field.endswith('/'):
                    key = field[0:-1]
                    self._dicts.append(key)
                else:
                    p = field.find('=')
                    if p > 0:
                        key = field[p + 1:]
                        resource = field[0:p] + '/openssh-key'
                    else:
                        key = resource = field
                    self._leaves[key] = resource
                self[key] = None

    def _materialize(self):
        for key in self:
            self[key]

    def __getitem__(self, key):
        if key not in self:
            return super(LazyLoadMetadata, self).__getitem__(key)
        val = super(LazyLoadMetadata, self).__getitem__(key)
        if val is not None:
            return val
        if key in self._leaves:
            resource = self._leaves[key]
            last_exception = None
            for i in range(0, self._num_retries):
                try:
                    val = boto.utils.retry_url(self._url + urllib.parse.quote(resource, safe='/:'), num_retries=self._num_retries, timeout=self._timeout)
                    if val and val[0] == '{':
                        val = json.loads(val)
                        break
                    else:
                        p = val.find('\n')
                        if p > 0:
                            val = val.split('\n')
                        break
                except JSONDecodeError as e:
                    boto.log.debug("encountered '%s' exception: %s" % (e.__class__.__name__, e))
                    boto.log.debug('corrupted JSON data found: %s' % val)
                    last_exception = e
                except Exception as e:
                    boto.log.debug('encountered unretryable' + " '%s' exception, re-raising" % e.__class__.__name__)
                    last_exception = e
                    raise
                boto.log.error('Caught exception reading meta data' + " for the '%s' try" % (i + 1))
                if i + 1 != self._num_retries:
                    next_sleep = min(random.random() * 2 ** i, boto.config.get('Boto', 'max_retry_delay', 60))
                    time.sleep(next_sleep)
            else:
                boto.log.error('Unable to read meta data, giving up')
                boto.log.error("encountered '%s' exception: %s" % (last_exception.__class__.__name__, last_exception))
                raise last_exception
            self[key] = val
        elif key in self._dicts:
            self[key] = LazyLoadMetadata(self._url + key + '/', self._num_retries)
        return super(LazyLoadMetadata, self).__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def values(self):
        self._materialize()
        return super(LazyLoadMetadata, self).values()

    def items(self):
        self._materialize()
        return super(LazyLoadMetadata, self).items()

    def __str__(self):
        self._materialize()
        return super(LazyLoadMetadata, self).__str__()

    def __repr__(self):
        self._materialize()
        return super(LazyLoadMetadata, self).__repr__()