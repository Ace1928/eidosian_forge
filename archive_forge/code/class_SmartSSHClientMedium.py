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
class SmartSSHClientMedium(SmartClientStreamMedium):
    """A client medium using SSH.

    It delegates IO to a SmartSimplePipesClientMedium or
    SmartClientAlreadyConnectedSocketMedium (depending on platform).
    """

    def __init__(self, base, ssh_params, vendor=None):
        """Creates a client that will connect on the first use.

        :param ssh_params: A SSHParams instance.
        :param vendor: An optional override for the ssh vendor to use. See
            breezy.transport.ssh for details on ssh vendors.
        """
        self._real_medium = None
        self._ssh_params = ssh_params
        self._scheme = 'bzr+ssh'
        SmartClientStreamMedium.__init__(self, base)
        self._vendor = vendor
        self._ssh_connection = None

    def __repr__(self):
        if self._ssh_params.port is None:
            maybe_port = ''
        else:
            maybe_port = ':%s' % self._ssh_params.port
        if self._ssh_params.username is None:
            maybe_user = ''
        else:
            maybe_user = '%s@' % self._ssh_params.username
        return '{}({}://{}{}{}/)'.format(self.__class__.__name__, self._scheme, maybe_user, self._ssh_params.host, maybe_port)

    def _accept_bytes(self, bytes):
        """See SmartClientStreamMedium.accept_bytes."""
        self._ensure_connection()
        self._real_medium.accept_bytes(bytes)

    def disconnect(self):
        """See SmartClientMedium.disconnect()."""
        if self._real_medium is not None:
            self._real_medium.disconnect()
            self._real_medium = None
        if self._ssh_connection is not None:
            self._ssh_connection.close()
            self._ssh_connection = None

    def _ensure_connection(self):
        """Connect this medium if not already connected."""
        if self._real_medium is not None:
            return
        if self._vendor is None:
            vendor = ssh._get_ssh_vendor()
        else:
            vendor = self._vendor
        self._ssh_connection = vendor.connect_ssh(self._ssh_params.username, self._ssh_params.password, self._ssh_params.host, self._ssh_params.port, command=[self._ssh_params.bzr_remote_path, 'serve', '--inet', '--directory=/', '--allow-writes'])
        io_kind, io_object = self._ssh_connection.get_sock_or_pipes()
        if io_kind == 'socket':
            self._real_medium = SmartClientAlreadyConnectedSocketMedium(self.base, io_object)
        elif io_kind == 'pipes':
            read_from, write_to = io_object
            self._real_medium = SmartSimplePipesClientMedium(read_from, write_to, self.base)
        else:
            raise AssertionError('Unexpected io_kind %r from %r' % (io_kind, self._ssh_connection))
        for hook in transport.Transport.hooks['post_connect']:
            hook(self)

    def _flush(self):
        """See SmartClientStreamMedium._flush()."""
        self._real_medium._flush()

    def _read_bytes(self, count):
        """See SmartClientStreamMedium.read_bytes."""
        if self._real_medium is None:
            raise errors.MediumNotConnected(self)
        return self._real_medium.read_bytes(count)