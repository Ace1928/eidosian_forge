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
def _separate_http_proxy_auth(self):
    url = self._proxy_url
    protocol = url[:url.find(':') + 3]
    no_protocol = url[len(protocol):]
    auth = no_protocol[:no_protocol.find('@')]
    proxy_without_auth = protocol + no_protocol[len(auth) + 1:]
    return (proxy_without_auth, auth)