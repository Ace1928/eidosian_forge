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
def _download_git(self, url, filename):
    filename = filename.split('#', 1)[0]
    url, rev = self._vcs_split_rev_from_url(url, pop_prefix=True)
    self.info('Doing git clone from %s to %s', url, filename)
    os.system('git clone --quiet %s %s' % (url, filename))
    if rev is not None:
        self.info('Checking out %s', rev)
        os.system('git -C %s checkout --quiet %s' % (filename, rev))
    return filename