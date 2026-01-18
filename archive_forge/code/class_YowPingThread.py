import time
import logging
from threading import Thread, Lock
from yowsup.layers import YowProtocolLayer, YowLayerEvent, EventCallback
from yowsup.common import YowConstants
from yowsup.layers.network import YowNetworkLayer
from yowsup.layers.auth import YowAuthenticationProtocolLayer
from .protocolentities import *
class YowPingThread(Thread):

    def __init__(self, layer, interval):
        assert type(layer) is YowIqProtocolLayer, 'layer must be a YowIqProtocolLayer, got %s instead.' % type(layer)
        self._layer = layer
        self._interval = interval
        self._stop = False
        self.__logger = logging.getLogger(__name__)
        super(YowPingThread, self).__init__()
        self.daemon = True
        self.name = 'YowPing%s' % self.name

    def run(self):
        while not self._stop:
            for i in range(0, self._interval):
                time.sleep(1)
                if self._stop:
                    self.__logger.debug('%s - ping thread stopped' % self.name)
                    return
            ping = PingIqProtocolEntity()
            self._layer.waitPong(ping.getId())
            if not self._stop:
                self._layer.sendIq(ping)

    def stop(self):
        self._stop = True