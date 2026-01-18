from __future__ import annotations
import atexit
import os
import sys
import debugpy
from debugpy import adapter, common, launcher
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import clients, components, launchers, servers, sessions
def _propagate_deferred_events(self):
    log.debug('Propagating deferred events to {0}...', self.client)
    for event in self._deferred_events:
        log.debug('Propagating deferred {0}', event.describe())
        self.client.channel.propagate(event)
    log.info('All deferred events propagated to {0}.', self.client)
    self._deferred_events = None