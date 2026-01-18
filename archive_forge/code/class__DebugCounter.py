import _thread
import errno
import io
import os
import sys
import time
import breezy
from ...lazy_import import lazy_import
import select
import socket
import weakref
from breezy import (
from breezy.i18n import gettext
from breezy.bzr.smart import client, protocol, request, signals, vfs
from breezy.transport import ssh
from ... import errors, osutils
class _DebugCounter:
    """An object that counts the HPSS calls made to each client medium.

    When a medium is garbage-collected, or failing that when
    breezy.global_state exits, the total number of calls made on that medium
    are reported via trace.note.
    """

    def __init__(self):
        self.counts = weakref.WeakKeyDictionary()
        client._SmartClient.hooks.install_named_hook('call', self.increment_call_count, 'hpss call counter')
        breezy.get_global_state().exit_stack.callback(self.flush_all)

    def track(self, medium):
        """Start tracking calls made to a medium.

        This only keeps a weakref to the medium, so shouldn't affect the
        medium's lifetime.
        """
        medium_repr = repr(medium)
        self.counts[medium] = dict(count=0, vfs_count=0, medium_repr=medium_repr)
        ref = weakref.ref(medium, self.done)

    def increment_call_count(self, params):
        value = self.counts[params.medium]
        value['count'] += 1
        try:
            request_method = request.request_handlers.get(params.method)
        except KeyError:
            return
        if issubclass(request_method, vfs.VfsRequest):
            value['vfs_count'] += 1

    def done(self, ref):
        value = self.counts[ref]
        count, vfs_count, medium_repr = (value['count'], value['vfs_count'], value['medium_repr'])
        value['count'] = 0
        value['vfs_count'] = 0
        if count != 0:
            trace.note(gettext('HPSS calls: {0} ({1} vfs) {2}').format(count, vfs_count, medium_repr))

    def flush_all(self):
        for ref in list(self.counts.keys()):
            self.done(ref)