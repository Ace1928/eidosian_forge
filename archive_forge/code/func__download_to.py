import sys
import os
import re
import io
import shutil
import socket
import base64
import hashlib
import itertools
import configparser
import html
import http.client
import urllib.parse
import urllib.request
import urllib.error
from functools import wraps
import setuptools
from pkg_resources import (
from distutils import log
from distutils.errors import DistutilsError
from fnmatch import translate
from setuptools.wheel import Wheel
from setuptools.extern.more_itertools import unique_everseen
def _download_to(self, url, filename):
    self.info('Downloading %s', url)
    fp = None
    try:
        checker = HashChecker.from_url(url)
        fp = self.open_url(url)
        if isinstance(fp, urllib.error.HTTPError):
            raise DistutilsError("Can't download %s: %s %s" % (url, fp.code, fp.msg))
        headers = fp.info()
        blocknum = 0
        bs = self.dl_blocksize
        size = -1
        if 'content-length' in headers:
            sizes = headers.get_all('Content-Length')
            size = max(map(int, sizes))
            self.reporthook(url, filename, blocknum, bs, size)
        with open(filename, 'wb') as tfp:
            while True:
                block = fp.read(bs)
                if block:
                    checker.feed(block)
                    tfp.write(block)
                    blocknum += 1
                    self.reporthook(url, filename, blocknum, bs, size)
                else:
                    break
            self.check_hash(checker, filename, tfp)
        return headers
    finally:
        if fp:
            fp.close()