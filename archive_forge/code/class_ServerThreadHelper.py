import threading
from oslo_config import cfg
from oslo_context.context import RequestContext
from oslo_utils import eventletutils
from oslotest import base
class ServerThreadHelper(threading.Thread):

    def __init__(self, server):
        super(ServerThreadHelper, self).__init__()
        self.daemon = True
        self._server = server
        self._stop_event = eventletutils.Event()
        self._start_event = eventletutils.Event()

    def start(self):
        super(ServerThreadHelper, self).start()
        self._start_event.wait()

    def run(self):
        self._server.start()
        self._start_event.set()
        self._stop_event.wait()
        self._server.start()
        self._server.stop()
        self._server.wait()

    def stop(self):
        self._stop_event.set()