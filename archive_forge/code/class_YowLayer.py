import unittest
import inspect
import threading
class YowLayer(object):
    __upper = None
    __lower = None
    _props = {}
    __detachedQueue = Queue.Queue()

    def __init__(self):
        self.setLayers(None, None)
        self.interface = None
        self.event_callbacks = {}
        self.__stack = None
        self.lock = threading.Lock()
        members = inspect.getmembers(self, predicate=inspect.ismethod)
        for m in members:
            if hasattr(m[1], 'event_callback'):
                fname = m[0]
                fn = m[1]
                self.event_callbacks[fn.event_callback] = getattr(self, fname)

    def getLayerInterface(self, YowLayerClass=None):
        return self.interface if YowLayerClass is None else self.__stack.getLayerInterface(YowLayerClass)

    def setStack(self, stack):
        self.__stack = stack

    def getStack(self):
        return self.__stack

    def setLayers(self, upper, lower):
        self.__upper = upper
        self.__lower = lower

    def send(self, data):
        self.toLower(data)

    def receive(self, data):
        self.toUpper(data)

    def toUpper(self, data):
        if self.__upper:
            self.__upper.receive(data)

    def toLower(self, data):
        self.lock.acquire()
        if self.__lower:
            self.__lower.send(data)
        self.lock.release()

    def emitEvent(self, yowLayerEvent):
        if self.__upper and (not self.__upper.onEvent(yowLayerEvent)):
            if yowLayerEvent.isDetached():
                yowLayerEvent.detached = False
                self.getStack().execDetached(lambda: self.__upper.emitEvent(yowLayerEvent))
            else:
                self.__upper.emitEvent(yowLayerEvent)

    def broadcastEvent(self, yowLayerEvent):
        if self.__lower and (not self.__lower.onEvent(yowLayerEvent)):
            if yowLayerEvent.isDetached():
                yowLayerEvent.detached = False
                self.getStack().execDetached(lambda: self.__lower.broadcastEvent(yowLayerEvent))
            else:
                self.__lower.broadcastEvent(yowLayerEvent)
    'return true to stop propagating the event'

    def onEvent(self, yowLayerEvent):
        eventName = yowLayerEvent.getName()
        if eventName in self.event_callbacks:
            return self.event_callbacks[eventName](yowLayerEvent)
        return False

    def getProp(self, key, default=None):
        return self.getStack().getProp(key, default)

    def setProp(self, key, val):
        return self.getStack().setProp(key, val)