import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
class ServerComm(threading.Thread):

    def __init__(self, notifications_queue, port, daemon=False):
        threading.Thread.__init__(self)
        self.setDaemon(daemon)
        self.finished = False
        self.notifications_queue = notifications_queue
        from _pydev_bundle import pydev_localhost
        encoding = file_system_encoding
        if encoding == 'mbcs':
            encoding = 'ISO-8859-1'
        self.server = xmlrpclib.Server('http://%s:%s' % (pydev_localhost.get_localhost(), port), encoding=encoding)

    def run(self):
        while True:
            kill_found = False
            commands = []
            command = self.notifications_queue.get(block=True)
            if isinstance(command, KillServer):
                kill_found = True
            else:
                assert isinstance(command, ParallelNotification)
                commands.append(command.to_tuple())
            try:
                while True:
                    command = self.notifications_queue.get(block=False)
                    if isinstance(command, KillServer):
                        kill_found = True
                    else:
                        assert isinstance(command, ParallelNotification)
                        commands.append(command.to_tuple())
            except:
                pass
            if commands:
                try:
                    self.server.notifyCommands(commands)
                except:
                    traceback.print_exc()
            if kill_found:
                self.finished = True
                return