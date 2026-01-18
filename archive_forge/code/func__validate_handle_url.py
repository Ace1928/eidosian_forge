from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients.os import swift
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _validate_handle_url(self):
    parts = self.url.path.split('/')
    msg = _('"%(url)s" is not a valid SwiftSignalHandle.  The %(part)s is invalid')
    cplugin = self.client_plugin()
    if not cplugin.is_valid_temp_url_path(self.url.path):
        raise ValueError(msg % {'url': self.url.path, 'part': 'Swift TempURL path'})
    if not parts[3] == self.stack.id:
        raise ValueError(msg % {'url': self.url.path, 'part': 'container name'})