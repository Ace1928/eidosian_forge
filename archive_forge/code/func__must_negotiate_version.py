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
def _must_negotiate_version(self):
    return self.api_version_select_state == 'user' and (self.os_ironic_api_version == 'latest' or isinstance(self.os_ironic_api_version, list))