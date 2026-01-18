from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
import multiprocessing
import os
import signal
import socket
import stat
import sys
import textwrap
import time
import webbrowser
from six.moves import input
from six.moves.http_client import ResponseNotReady
import boto
from boto.provider import Provider
import gslib
from gslib.command import Command
from gslib.command import DEFAULT_TASK_ESTIMATION_THRESHOLD
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.cred_types import CredTypes
from gslib.exception import AbortException
from gslib.exception import CommandException
from gslib.metrics import CheckAndMaybePromptForAnalyticsEnabling
from gslib.sig_handling import RegisterSignalHandler
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils.hashing_helper import CHECK_HASH_ALWAYS
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_FAIL
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_SKIP
from gslib.utils.hashing_helper import CHECK_HASH_NEVER
from gslib.utils.parallelism_framework_util import ShouldProhibitMultiprocessing
from httplib2 import ServerNotFoundError
from oauth2client.client import HAS_CRYPTO
def _WriteProxyConfigFileSection(self, config_file):
    """Writes proxy section of configuration file.

    Args:
      config_file: File object to which the resulting config file will be
          written.
    """
    config = boto.config
    config_file.write('# To use a proxy, edit and uncomment the proxy and proxy_port lines.\n# If you need a user/password with this proxy, edit and uncomment\n# those lines as well. If your organization also disallows DNS\n# lookups by client machines, set proxy_rdns to True (the default).\n# If you have installed gsutil through the Cloud SDK and have \n# configured proxy settings in gcloud, those proxy settings will \n# override any other options (including those set here, along with \n# any settings in proxy-related environment variables). Otherwise, \n# if proxy_host and proxy_port are not specified in this file and\n# one of the OS environment variables http_proxy, https_proxy, or\n# HTTPS_PROXY is defined, gsutil will use the proxy server specified\n# in these environment variables, in order of precedence according\n# to how they are listed above.\n')
    self._WriteConfigLineMaybeCommented(config_file, 'proxy', config.get_value('Boto', 'proxy', None), 'proxy host')
    self._WriteConfigLineMaybeCommented(config_file, 'proxy_type', config.get_value('Boto', 'proxy_type', None), 'proxy type (socks4, socks5, http) | Defaults to http')
    self._WriteConfigLineMaybeCommented(config_file, 'proxy_port', config.get_value('Boto', 'proxy_port', None), 'proxy port')
    self._WriteConfigLineMaybeCommented(config_file, 'proxy_user', config.get_value('Boto', 'proxy_user', None), 'proxy user')
    self._WriteConfigLineMaybeCommented(config_file, 'proxy_pass', config.get_value('Boto', 'proxy_pass', None), 'proxy password')
    self._WriteConfigLineMaybeCommented(config_file, 'proxy_rdns', config.get_value('Boto', 'proxy_rdns', False), 'let proxy server perform DNS lookups (True,False); socks proxy not supported')