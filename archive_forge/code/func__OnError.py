from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
import ssl
import sys
import threading
import traceback
from googlecloudsdk.api_lib.compute import iap_tunnel_lightweight_websocket as iap_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
import six
import websocket
def _OnError(self, exception_obj):
    if not self._is_closed:
        log.debug('[%d] Error during WebSocket processing.', self._conn_id, exc_info=True)
        log.info('[%d] Error during WebSocket processing:\n' + ''.join(traceback.format_exception_only(type(exception_obj), exception_obj)), self._conn_id)
        self._error_msg = six.text_type(exception_obj)