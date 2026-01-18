import threading
import fasteners
from oslo_utils import excutils
from taskflow import flow
from taskflow import logging
from taskflow import task
from taskflow.types import graph as gr
from taskflow.types import tree as tr
from taskflow.utils import iter_utils
from taskflow.utils import misc
from taskflow.flow import (LINK_INVARIANT, LINK_RETRY)  # noqa
def _post_item_compile(self, item, graph, node):
    """Called after a item is compiled; doing post-compilation actions."""
    self._level -= 1
    if LOG.isEnabledFor(logging.TRACE):
        prefix = '  ' * self._level
        LOG.trace("%sDecomposed '%s' into:", prefix, item)
        prefix = '  ' * (self._level + 1)
        LOG.trace('%sGraph:', prefix)
        for line in graph.pformat().splitlines():
            LOG.trace('%s  %s', prefix, line)
        LOG.trace('%sHierarchy:', prefix)
        for line in node.pformat().splitlines():
            LOG.trace('%s  %s', prefix, line)