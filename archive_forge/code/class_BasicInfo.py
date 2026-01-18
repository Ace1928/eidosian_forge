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
class BasicInfo(object):
    """Holds basic information about your system setup."""

    def __init__(self, anonymizer=None):
        anonymizer = anonymizer or NoopAnonymizer()
        platform = platforms.Platform.Current()
        self.version = config.CLOUD_SDK_VERSION
        self.operating_system = platform.operating_system
        self.architecture = platform.architecture
        self.python_location = anonymizer.ProcessPath(sys.executable and encoding.Decode(sys.executable))
        self.python_version = sys.version
        self.default_ca_certs_file = anonymizer.ProcessPath(certifi.where())
        self.site_packages = 'site' in sys.modules
        self.locale = self._GetDefaultLocale()

    def __str__(self):
        return textwrap.dedent('        Google Cloud SDK [{version}]\n\n        Platform: [{os}, {arch}] {uname}\n        Locale: {locale}\n        Python Version: [{python_version}]\n        Python Location: [{python_location}]\n        OpenSSL: [{openssl_version}]\n        Requests Version: [{requests_version}]\n        urllib3 Version: [{urllib3_version}]\n        Default CA certs file: [{default_ca_certs_file}]\n        Site Packages: [{site_packages}]\n        '.format(version=self.version, os=self.operating_system.name if self.operating_system else 'unknown', arch=self.architecture.name if self.architecture else 'unknown', uname=system_platform.uname(), locale=self.locale, python_location=self.python_location, python_version=self.python_version.replace('\n', ' '), openssl_version=ssl.OPENSSL_VERSION, requests_version=requests.__version__, urllib3_version=urllib3.__version__, default_ca_certs_file=self.default_ca_certs_file, site_packages='Enabled' if self.site_packages else 'Disabled'))

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