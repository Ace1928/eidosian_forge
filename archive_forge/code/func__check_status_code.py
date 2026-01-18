import importlib
import logging
import os
import sys
import warnings
from keystoneauth1 import adapter
from keystoneauth1 import session as ks_session
from oslo_utils import importutils
from barbicanclient import exceptions
def _check_status_code(self, resp):
    status = resp.status_code
    LOG.debug('Response status {0}'.format(status))
    if status == 401:
        LOG.error('Auth error: {0}'.format(self._get_error_message(resp)))
        raise exceptions.HTTPAuthError('{0}'.format(self._get_error_message(resp)))
    if not status or status >= 500:
        LOG.error('5xx Server error: {0}'.format(self._get_error_message(resp)))
        raise exceptions.HTTPServerError('{0}'.format(self._get_error_message(resp)), status)
    if status >= 400:
        LOG.error('4xx Client error: {0}'.format(self._get_error_message(resp)))
        raise exceptions.HTTPClientError('{0}'.format(self._get_error_message(resp)), status)