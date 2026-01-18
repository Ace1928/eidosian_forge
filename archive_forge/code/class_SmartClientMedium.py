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
class SmartClientMedium(SmartMedium):
    """Smart client is a medium for sending smart protocol requests over."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self._protocol_version_error = None
        self._protocol_version = None
        self._done_hello = False
        self._remote_version_is_before = None
        if 'hpss' in debug.debug_flags:
            global _debug_counter
            if _debug_counter is None:
                _debug_counter = _DebugCounter()
            _debug_counter.track(self)
        if 'hpss_client_no_vfs' in debug.debug_flags:
            global _vfs_refuser
            if _vfs_refuser is None:
                _vfs_refuser = _VfsRefuser()

    def _is_remote_before(self, version_tuple):
        """Is it possible the remote side supports RPCs for a given version?

        Typical use::

            needed_version = (1, 2)
            if medium._is_remote_before(needed_version):
                fallback_to_pre_1_2_rpc()
            else:
                try:
                    do_1_2_rpc()
                except UnknownSmartMethod:
                    medium._remember_remote_is_before(needed_version)
                    fallback_to_pre_1_2_rpc()

        :seealso: _remember_remote_is_before
        """
        if self._remote_version_is_before is None:
            return False
        return version_tuple >= self._remote_version_is_before

    def _remember_remote_is_before(self, version_tuple):
        """Tell this medium that the remote side is older the given version.

        :seealso: _is_remote_before
        """
        if self._remote_version_is_before is not None and version_tuple > self._remote_version_is_before:
            trace.mutter('_remember_remote_is_before(%r) called, but _remember_remote_is_before(%r) was called previously.', version_tuple, self._remote_version_is_before)
            if 'hpss' in debug.debug_flags:
                ui.ui_factory.show_warning('_remember_remote_is_before(%r) called, but _remember_remote_is_before(%r) was called previously.' % (version_tuple, self._remote_version_is_before))
            return
        self._remote_version_is_before = version_tuple

    def protocol_version(self):
        """Find out if 'hello' smart request works."""
        if self._protocol_version_error is not None:
            raise self._protocol_version_error
        if not self._done_hello:
            try:
                medium_request = self.get_request()
                client_protocol = protocol.SmartClientRequestProtocolOne(medium_request)
                client_protocol.query_version()
                self._done_hello = True
            except errors.SmartProtocolError as e:
                self._protocol_version_error = e
                raise
        return '2'

    def should_probe(self):
        """Should RemoteBzrDirFormat.probe_transport send a smart request on
        this medium?

        Some transports are unambiguously smart-only; there's no need to check
        if the transport is able to carry smart requests, because that's all
        it is for.  In those cases, this method should return False.

        But some HTTP transports can sometimes fail to carry smart requests,
        but still be usuable for accessing remote bzrdirs via plain file
        accesses.  So for those transports, their media should return True here
        so that RemoteBzrDirFormat can determine if it is appropriate for that
        transport.
        """
        return False

    def disconnect(self):
        """If this medium maintains a persistent connection, close it.

        The default implementation does nothing.
        """

    def remote_path_from_transport(self, transport):
        """Convert transport into a path suitable for using in a request.

        Note that the resulting remote path doesn't encode the host name or
        anything but path, so it is only safe to use it in requests sent over
        the medium from the matching transport.
        """
        medium_base = urlutils.join(self.base, '/')
        rel_url = urlutils.relative_url(medium_base, transport.base)
        return urlutils.unquote(rel_url)