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
def _download_url(self, scheme, url, tmpdir):
    name, fragment = egg_info_for_url(url)
    if name:
        while '..' in name:
            name = name.replace('..', '.').replace('\\', '_')
    else:
        name = '__downloaded__'
    if name.endswith('.egg.zip'):
        name = name[:-4]
    filename = os.path.join(tmpdir, name)
    if scheme == 'svn' or scheme.startswith('svn+'):
        return self._download_svn(url, filename)
    elif scheme == 'git' or scheme.startswith('git+'):
        return self._download_git(url, filename)
    elif scheme.startswith('hg+'):
        return self._download_hg(url, filename)
    elif scheme == 'file':
        return urllib.request.url2pathname(urllib.parse.urlparse(url)[2])
    else:
        self.url_ok(url, True)
        return self._attempt_download(url, filename)