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
def add_find_links(self, urls):
    """Add `urls` to the list that will be prescanned for searches"""
    for url in urls:
        if self.to_scan is None or not URL_SCHEME(url) or url.startswith('file:') or list(distros_for_url(url)):
            self.scan_url(url)
        else:
            self.to_scan.append(url)