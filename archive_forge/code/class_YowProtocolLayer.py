import unittest
import inspect
import threading
class YowProtocolLayer(YowLayer):

    def __init__(self, handleMap=None):
        super(YowProtocolLayer, self).__init__()
        self.handleMap = handleMap or {}
        self.iqRegistry = {}

    def receive(self, node):
        if not self.processIqRegistry(node):
            if node.tag in self.handleMap:
                recv, _ = self.handleMap[node.tag]
                if recv:
                    recv(node)

    def send(self, entity):
        if entity.getTag() in self.handleMap:
            _, send = self.handleMap[entity.getTag()]
            if send:
                send(entity)

    def entityToLower(self, entity):
        self.toLower(entity.toProtocolTreeNode())

    def isGroupJid(self, jid):
        return '-' in jid

    def raiseErrorForNode(self, node):
        raise ValueError('Unimplemented notification type %s ' % node)

    def _sendIq(self, iqEntity, onSuccess=None, onError=None):
        self.iqRegistry[iqEntity.getId()] = (iqEntity, onSuccess, onError)
        self.toLower(iqEntity.toProtocolTreeNode())

    def processIqRegistry(self, protocolTreeNode):
        if protocolTreeNode.tag == 'iq':
            iq_id = protocolTreeNode['id']
            if iq_id in self.iqRegistry:
                originalIq, successClbk, errorClbk = self.iqRegistry[iq_id]
                del self.iqRegistry[iq_id]
                if protocolTreeNode['type'] == 'result' and successClbk:
                    successClbk(protocolTreeNode, originalIq)
                elif protocolTreeNode['type'] == 'error' and errorClbk:
                    errorClbk(protocolTreeNode, originalIq)
                return True
        return False