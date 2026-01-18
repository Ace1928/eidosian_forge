from yowsup.structs import ProtocolTreeNode
from yowsup.layers.protocol_iq.protocolentities import IqProtocolEntity
from .iq_sync import SyncIqProtocolEntity
class GetSyncIqProtocolEntity(SyncIqProtocolEntity):
    MODE_FULL = 'full'
    MODE_DELTA = 'delta'
    CONTEXT_REGISTRATION = 'registration'
    CONTEXT_INTERACTIVE = 'interactive'
    CONTEXTS = (CONTEXT_REGISTRATION, CONTEXT_INTERACTIVE)
    MODES = (MODE_FULL, MODE_DELTA)
    '\n    <iq type="get" id="{{id}}" xmlns="urn:xmpp:whatsapp:sync">\n        <sync mode="{{full | ?}}"\n            context="{{registration | ?}}"\n            sid="{{str((int(time.time()) + 11644477200) * 10000000)}}"\n            index="{{0 | ?}}"\n            last="{{true | false?}}"\n        >\n            <user>\n                {{num1}}\n            </user>\n            <user>\n                {{num2}}\n            </user>\n\n        </sync>\n    </iq>\n    '

    def __init__(self, numbers, mode=MODE_FULL, context=CONTEXT_INTERACTIVE, sid=None, index=0, last=True):
        super(GetSyncIqProtocolEntity, self).__init__('get', sid=sid, index=index, last=last)
        self.setGetSyncProps(numbers, mode, context)

    def setGetSyncProps(self, numbers, mode, context):
        assert type(numbers) is list, 'numbers must be a list'
        assert mode in self.__class__.MODES, 'mode must be in %s' % self.__class__.MODES
        assert context in self.__class__.CONTEXTS, 'context must be in %s' % self.__class__.CONTEXTS
        self.numbers = numbers
        self.mode = mode
        self.context = context

    def __str__(self):
        out = super(GetSyncIqProtocolEntity, self).__str__()
        out += 'Mode: %s\n' % self.mode
        out += 'Context: %s\n' % self.context
        out += 'numbers: %s\n' % ','.join(self.numbers)
        return out

    def toProtocolTreeNode(self):
        users = [ProtocolTreeNode('user', {}, None, number.encode()) for number in self.numbers]
        node = super(GetSyncIqProtocolEntity, self).toProtocolTreeNode()
        syncNode = node.getChild('sync')
        syncNode.setAttribute('mode', self.mode)
        syncNode.setAttribute('context', self.context)
        syncNode.addChildren(users)
        return node

    @staticmethod
    def fromProtocolTreeNode(node):
        syncNode = node.getChild('sync')
        userNodes = syncNode.getAllChildren()
        numbers = [userNode.data.decode() for userNode in userNodes]
        entity = SyncIqProtocolEntity.fromProtocolTreeNode(node)
        entity.__class__ = GetSyncIqProtocolEntity
        entity.setGetSyncProps(numbers, syncNode.getAttributeValue('mode'), syncNode.getAttributeValue('context'))
        return entity