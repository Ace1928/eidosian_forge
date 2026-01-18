import unittest
import inspect
import threading
class YowProtocolLayerTest(YowLayerTest):

    def assertSent(self, entity):
        self.send(entity)
        try:
            self.assertEqual(entity.toProtocolTreeNode(), self.lowerSink.pop())
        except IndexError:
            raise AssertionError("Entity '%s' was not sent through this layer" % entity.getTag())

    def assertReceived(self, entity):
        node = entity.toProtocolTreeNode()
        self.receive(node)
        try:
            self.assertEqual(node, self.upperSink.pop().toProtocolTreeNode())
        except IndexError:
            raise AssertionError("'%s' was not received through this layer" % entity.getTag())