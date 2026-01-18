import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
def _is_current_session_active(self):
    try:
        self._send_request('get', 'https', self.CURRENT_SESSIONS_PATH, headers=self._get_auth_header())
        LOG.debug('Current session %s is active.', self._trunc_session_id())
        return True
    except (exc.LogInsightLoginTimeout, exc.LogInsightAPIError):
        LOG.debug('Current session %s is not active.', self._trunc_session_id())
        return False