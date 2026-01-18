import collections
import contextlib
import functools
import sys
import threading
import fasteners
import futurist
from kazoo import exceptions as k_exceptions
from kazoo.protocol import paths as k_paths
from kazoo.protocol import states as k_states
from kazoo.recipe import watchers
from oslo_serialization import jsonutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow.conductors import base as c_base
from taskflow import exceptions as excp
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import kazoo_utils
from taskflow.utils import misc
def _state_change_listener(self, state):
    if self._last_states:
        LOG.debug("Kazoo client has changed to state '%s' from prior states '%s'", state, self._last_states)
    else:
        LOG.debug("Kazoo client has changed to state '%s' (from its initial/uninitialized state)", state)
    self._last_states.appendleft(state)
    if state == k_states.KazooState.LOST:
        self._connected = False
        if not self._closing:
            LOG.warning('Connection to zookeeper has been lost')
    elif state == k_states.KazooState.SUSPENDED:
        LOG.warning('Connection to zookeeper has been suspended')
        self._suspended = True
    elif self._suspended:
        self._suspended = False