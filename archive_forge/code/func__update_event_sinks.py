import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
def _update_event_sinks(self, sinks):
    self._event_sinks.extend(sinks)
    for sink in sinks:
        sink = sink.copy()
        sink_class = sink.pop('type')
        sink_class = self.event_sink_classes[sink_class]
        self._built_event_sinks.append(sink_class(**sink))