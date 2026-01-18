from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import getpass
import io
import locale
import os
import platform as system_platform
import re
import ssl
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import http_proxy_setup
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import requests
import six
import urllib3
def _GetDefaultLocale(self):
    """Determines the locale from the program's environment.

    Returns:
      String: Default locale, with a fallback to locale environment variables.
    """
    env_vars = ['%s:%s' % (var, encoding.GetEncodedValue(os.environ, var)) for var in ['LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE'] if encoding.GetEncodedValue(os.environ, var)]
    fallback_locale = '; '.join(env_vars)
    try:
        return locale.getlocale()
    except ValueError:
        return fallback_locale