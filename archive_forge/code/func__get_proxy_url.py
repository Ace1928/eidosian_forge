import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
def _get_proxy_url(self):
    if self._url.startswith('https://'):
        return os.environ.get('https_proxy', os.environ.get('HTTPS_PROXY'))
    if self._url.startswith('http://'):
        return os.environ.get('http_proxy', os.environ.get('HTTP_PROXY'))