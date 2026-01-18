import logging
import ssl
import time
from oslo_utils import excutils
from oslo_utils import netutils
import requests
import urllib.parse as urlparse
from urllib3 import connection as httplib
from oslo_vmware._i18n import _
from oslo_vmware import exceptions
from oslo_vmware import vim_util
def _create_read_connection(self, url, cookies=None, cacerts=False, ssl_thumbprint=None):
    LOG.debug('Opening URL: %s for reading.', url)
    try:
        conn = self._create_connection(url, 'GET', cacerts, ssl_thumbprint, cookies=cookies)
        conn.endheaders()
        return conn
    except Exception as excep:
        excep_msg = _('Error occurred while opening URL: %s for reading.') % url
        LOG.exception(excep_msg)
        raise exceptions.VimException(excep_msg, excep)