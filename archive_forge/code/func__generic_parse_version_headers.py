from distutils.version import StrictVersion
import functools
from http import client as http_client
import json
import logging
import re
import textwrap
import time
from urllib import parse as urlparse
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common.i18n import _
from ironicclient import exc
def _generic_parse_version_headers(self, accessor_func):
    min_ver = accessor_func('X-OpenStack-Ironic-API-Minimum-Version', None)
    max_ver = accessor_func('X-OpenStack-Ironic-API-Maximum-Version', None)
    return (min_ver, max_ver)