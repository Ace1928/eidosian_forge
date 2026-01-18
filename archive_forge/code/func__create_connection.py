import os
import random
import stat
import time
from io import BytesIO
from urllib.parse import urlparse, urlunparse
from .. import config, debug, errors, osutils, ui, urlutils
from ..tests.test_server import TestServer
from ..trace import mutter
from . import (ConnectedTransport, FileExists, FileStream, NoSuchFile,
def _create_connection(self, credentials=None):
    if credentials is None:
        user, password = (self._parsed_url.user, self._parsed_url.password)
    else:
        user, password = credentials
    try:
        connection = gio.File(self.url)
        mount = None
        try:
            mount = connection.find_enclosing_mount()
        except gio.Error as e:
            if e.code == gio.ERROR_NOT_MOUNTED:
                self.loop = glib.MainLoop()
                ui.ui_factory.show_message('Mounting %s using GIO' % self.url)
                op = gio.MountOperation()
                if user:
                    op.set_username(user)
                if password:
                    op.set_password(password)
                op.connect('ask-password', self._auth_cb)
                m = connection.mount_enclosing_volume(op, self._mount_done_cb)
                self.loop.run()
    except gio.Error as e:
        raise errors.TransportError(msg='Error setting up connection: %s' % str(e), orig_error=e)
    return (connection, (user, password))