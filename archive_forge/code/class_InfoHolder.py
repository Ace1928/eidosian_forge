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
class InfoHolder(object):
    """Base object to hold all the configuration info."""

    def __init__(self, anonymizer=None):
        self.basic = BasicInfo(anonymizer)
        self.installation = InstallationInfo(anonymizer)
        self.config = ConfigInfo(anonymizer)
        self.env_proxy = ProxyInfoFromEnvironmentVars(anonymizer)
        self.logs = LogsInfo(anonymizer)
        self.tools = ToolsInfo(anonymizer)

    def __str__(self):
        out = io.StringIO()
        out.write(six.text_type(self.basic) + '\n')
        out.write(six.text_type(self.installation) + '\n')
        out.write(six.text_type(self.config) + '\n')
        if six.text_type(self.env_proxy):
            out.write(six.text_type(self.env_proxy) + '\n')
        out.write(six.text_type(self.logs) + '\n')
        out.write(six.text_type(self.tools) + '\n')
        return out.getvalue()