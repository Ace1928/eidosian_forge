import logging
import sys
from cliff import app
from cliff import commandmanager
import openstack
from openstack import config as os_config
from osc_lib import utils
import pbr.version
from ironicclient.common import http
from ironicclient.common.i18n import _
from ironicclient import exc
from ironicclient.v1 import client
def _configure_ironic_logging(self):
    debug_enabled = self.options.debug or self.options.verbose_level > 1
    openstack.enable_logging(debug=debug_enabled, stream=sys.stderr, format_template=self._STREAM_FORMAT, format_stream=debug_enabled)
    for name in ('openstack', 'keystoneauth'):
        logger = logging.getLogger(name)
        logger.propagate = False
    for name in ('ironicclient', 'ironic_inspector_client'):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if debug_enabled else logging.WARNING)
        if not logger.handlers and debug_enabled:
            handler = logging.StreamHandler(stream=sys.stderr)
            handler.setFormatter(logging.Formatter(self._STREAM_FORMAT))
            logger.addHandler(handler)
            logger.propagate = False