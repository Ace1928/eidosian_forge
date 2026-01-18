from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
import tempfile
import textwrap
import six
import boto
from boto import config
import boto.auth
from boto.exception import NoAuthHandlerFound
from boto.gs.connection import GSConnection
from boto.provider import Provider
from boto.pyami.config import BotoConfigLocations
import gslib
from gslib import context_config
from gslib.exception import CommandException
from gslib.utils import system_util
from gslib.utils.constants import DEFAULT_GCS_JSON_API_VERSION
from gslib.utils.constants import DEFAULT_GSUTIL_STATE_DIR
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import UTF8
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import ONE_MIB
import httplib2
from oauth2client.client import HAS_CRYPTO
def SetProxyInfo(boto_proxy_config):
    """Sets proxy info from boto and environment and converts to httplib2.ProxyInfo.

  Args:
    dict: Values read from the .boto file

  Returns:
    httplib2.ProxyInfo constructed from boto or environment variable string.
  """
    proxy_type_spec = {'socks4': 1, 'socks5': 2, 'http': 3, 'https': 3}
    proxy_type = proxy_type_spec.get(boto_proxy_config.get('proxy_type').lower(), proxy_type_spec['http'])
    proxy_host = boto_proxy_config.get('proxy_host')
    proxy_port = boto_proxy_config.get('proxy_port')
    proxy_user = boto_proxy_config.get('proxy_user')
    proxy_pass = boto_proxy_config.get('proxy_pass')
    proxy_rdns = bool(boto_proxy_config.get('proxy_rdns'))
    proxy_info = httplib2.ProxyInfo(proxy_host=proxy_host, proxy_type=proxy_type, proxy_port=proxy_port, proxy_user=proxy_user, proxy_pass=proxy_pass, proxy_rdns=proxy_rdns)
    if not proxy_info.proxy_type == proxy_type_spec['http']:
        proxy_info.proxy_rdns = False
    if not (proxy_info.proxy_host and proxy_info.proxy_port):
        for proxy_env_var in ['http_proxy', 'https_proxy', 'HTTPS_PROXY']:
            if proxy_env_var in os.environ and os.environ[proxy_env_var]:
                proxy_info = ProxyInfoFromEnvironmentVar(proxy_env_var)
                if boto_proxy_config.get('proxy_rdns') == None:
                    proxy_info.proxy_rdns = True
                break
    return proxy_info