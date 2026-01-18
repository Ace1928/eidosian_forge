import unittest
import inspect
import threading
class YowLayerTest(unittest.TestCase):

    def __init__(self, *args):
        super(YowLayerTest, self).__init__(*args)
        self.upperSink = []
        self.lowerSink = []
        self.toUpper = self.receiveOverrider
        self.toLower = self.sendOverrider
        self.upperEventSink = []
        self.lowerEventSink = []
        self.emitEvent = self.emitEventOverrider
        self.broadcastEvent = self.broadcastEventOverrider

    def receiveOverrider(self, data):
        self.upperSink.append(data)

    def sendOverrider(self, data):
        self.lowerSink.append(data)

    def emitEventOverrider(self, event):
        self.upperEventSink.append(event)

    def broadcastEventOverrider(self, event):
        self.lowerEventSink.append(event)

    def assert_emitEvent(self, event):
        self.emitEvent(event)
        try:
            self.assertEqual(event, self.upperEventSink.pop())
        except IndexError:
            raise AssertionError("Event '%s' was not emited through this layer" % event.getName())

    def assert_broadcastEvent(self, event):
        self.broadcastEvent(event)
        try:
            self.assertEqual(event, self.lowerEventSink.pop())
        except IndexError:
            raise AssertionError("Event '%s' was not broadcasted through this layer" % event.getName())