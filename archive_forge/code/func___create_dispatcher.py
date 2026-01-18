from yowsup.layers import YowLayer, YowLayerEvent, EventCallback
from yowsup.layers.network.layer_interface import YowNetworkLayerInterface
from yowsup.layers.network.dispatcher.dispatcher import ConnectionCallbacks
from yowsup.layers.network.dispatcher.dispatcher import YowConnectionDispatcher
from yowsup.layers.network.dispatcher.dispatcher_socket import SocketConnectionDispatcher
from yowsup.layers.network.dispatcher.dispatcher_asyncore import AsyncoreConnectionDispatcher
import logging
def __create_dispatcher(self, dispatcher_type):
    if dispatcher_type == self.DISPATCHER_ASYNCORE:
        logger.debug('Created asyncore dispatcher')
        return AsyncoreConnectionDispatcher(self)
    else:
        logger.debug('Created socket dispatcher')
        return SocketConnectionDispatcher(self)