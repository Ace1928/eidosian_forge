import os
import socket
from unittest import mock
from oslotest import base as test_base
from oslo_service import systemd
@mock.patch.object(os, 'getenv', return_value='@fake_socket')
def _test__sd_notify(self, getenv_mock, unset_env=False):
    self.ready = False
    self.closed = False

    class FakeSocket(object):

        def __init__(self, family, type):
            pass

        def connect(fs, socket):
            pass

        def close(fs):
            self.closed = True

        def sendall(fs, data):
            if data == b'READY=1':
                self.ready = True
    with mock.patch.object(socket, 'socket', new=FakeSocket):
        if unset_env:
            systemd.notify_once()
        else:
            systemd.notify()
        self.assertTrue(self.ready)
        self.assertTrue(self.closed)