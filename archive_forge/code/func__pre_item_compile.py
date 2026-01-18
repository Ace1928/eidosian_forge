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
def _pre_item_compile(self, item):
    """Called before a item is compiled; any pre-compilation actions."""
    if item in self._history:
        raise ValueError("Already compiled item '%s' (%s), duplicate and/or recursive compiling is not supported" % (item, type(item)))
    self._history.add(item)
    if LOG.isEnabledFor(logging.TRACE):
        LOG.trace("%sCompiling '%s'", '  ' * self._level, item)
    self._level += 1