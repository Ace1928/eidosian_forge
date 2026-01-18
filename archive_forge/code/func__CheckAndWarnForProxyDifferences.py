from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import errno
import getopt
import logging
import os
import re
import signal
import socket
import sys
import textwrap
import traceback
import six
from six.moves import configparser
from six.moves import range
from google.auth import exceptions as google_auth_exceptions
import gslib.exception
from gslib.exception import CommandException
from gslib.exception import ControlCException
from gslib.utils.version_check import check_python_version_support
from gslib.utils.arg_helper import GetArgumentsAndOptions
from gslib.utils.user_agent_helper import GetUserAgent
import boto
import gslib
from gslib.utils import system_util, text_util
from gslib import metrics
import httplib2
import oauth2client
from google_reauth import reauth_creds
from google_reauth import errors as reauth_errors
from gslib import context_config
from gslib import wildcard_iterator
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import ProjectIdException
from gslib.cloud_api import ServiceException
from gslib.command_runner import CommandRunner
import apitools.base.py.exceptions as apitools_exceptions
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import InitializeSignalHandling
from gslib.sig_handling import RegisterSignalHandler
def _CheckAndWarnForProxyDifferences():
    boto_port = boto.config.getint('Boto', 'proxy_port', 0)
    if boto.config.get('Boto', 'proxy', None) or boto_port:
        for proxy_env_var in ['http_proxy', 'https_proxy', 'HTTPS_PROXY']:
            if proxy_env_var in os.environ and os.environ[proxy_env_var]:
                differing_values = []
                proxy_info = boto_util.ProxyInfoFromEnvironmentVar(proxy_env_var)
                if proxy_info.proxy_host != boto.config.get('Boto', 'proxy', None):
                    differing_values.append('Boto proxy host: "%s" differs from %s proxy host: "%s"' % (boto.config.get('Boto', 'proxy', None), proxy_env_var, proxy_info.proxy_host))
                if proxy_info.proxy_user != boto.config.get('Boto', 'proxy_user', None):
                    differing_values.append('Boto proxy user: "%s" differs from %s proxy user: "%s"' % (boto.config.get('Boto', 'proxy_user', None), proxy_env_var, proxy_info.proxy_user))
                if proxy_info.proxy_pass != boto.config.get('Boto', 'proxy_pass', None):
                    differing_values.append('Boto proxy password differs from %s proxy password' % proxy_env_var)
                if (proxy_info.proxy_port or boto_port) and proxy_info.proxy_port != boto_port:
                    differing_values.append('Boto proxy port: "%s" differs from %s proxy port: "%s"' % (boto_port, proxy_env_var, proxy_info.proxy_port))
                if differing_values:
                    sys.stderr.write('\n'.join(textwrap.wrap('WARNING: Proxy configuration is present in both the %s environment variable and boto configuration, but configuration differs. boto configuration proxy values will be used. Differences detected:' % proxy_env_var)))
                    sys.stderr.write('\n%s\n' % '\n'.join(differing_values))
                del os.environ[proxy_env_var]