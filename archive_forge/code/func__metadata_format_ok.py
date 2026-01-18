import collections
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.resources import signal_responder
def _metadata_format_ok(self, metadata):
    if not isinstance(metadata, collections.abc.Mapping):
        return False
    if set(metadata) != set(self.METADATA_KEYS):
        return False
    return self._status_ok(metadata[self.STATUS])