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
def _post_compile(self, graph, node):
    """Called after the compilation of the root finishes successfully."""
    self._history.clear()
    self._level = 0