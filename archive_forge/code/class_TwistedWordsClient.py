from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
class TwistedWordsClient(pb.Referenceable, basesupport.AbstractClientMixin):
    """In some cases, this acts as an Account, since it a source of text
    messages (multiple Words instances may be on a single PB connection)
    """

    def __init__(self, acct, serviceName, perspectiveName, chatui, _logonDeferred=None):
        self.accountName = '{} ({}:{})'.format(acct.accountName, serviceName, perspectiveName)
        self.name = perspectiveName
        print('HELLO I AM A PB SERVICE', serviceName, perspectiveName)
        self.chat = chatui
        self.account = acct
        self._logonDeferred = _logonDeferred

    def getPerson(self, name):
        return self.chat.getPerson(name, self)

    def getGroup(self, name):
        return self.chat.getGroup(name, self)

    def getGroupConversation(self, name):
        return self.chat.getGroupConversation(self.getGroup(name))

    def addContact(self, name):
        self.perspective.callRemote('addContact', name)

    def remote_receiveGroupMembers(self, names, group):
        print('received group members:', names, group)
        self.getGroupConversation(group).setGroupMembers(names)

    def remote_receiveGroupMessage(self, sender, group, message, metadata=None):
        print('received a group message', sender, group, message, metadata)
        self.getGroupConversation(group).showGroupMessage(sender, message, metadata)

    def remote_memberJoined(self, member, group):
        print('member joined', member, group)
        self.getGroupConversation(group).memberJoined(member)

    def remote_memberLeft(self, member, group):
        print('member left')
        self.getGroupConversation(group).memberLeft(member)

    def remote_notifyStatusChanged(self, name, status):
        self.chat.getPerson(name, self).setStatus(status)

    def remote_receiveDirectMessage(self, name, message, metadata=None):
        self.chat.getConversation(self.chat.getPerson(name, self)).showMessage(message, metadata)

    def remote_receiveContactList(self, clist):
        for name, status in clist:
            self.chat.getPerson(name, self).setStatus(status)

    def remote_setGroupMetadata(self, dict_, groupName):
        if 'topic' in dict_:
            self.getGroupConversation(groupName).setTopic(dict_['topic'], dict_.get('topic_author', None))

    def joinGroup(self, name):
        self.getGroup(name).joining()
        return self.perspective.callRemote('joinGroup', name).addCallback(self._cbGroupJoined, name)

    def leaveGroup(self, name):
        self.getGroup(name).leaving()
        return self.perspective.callRemote('leaveGroup', name).addCallback(self._cbGroupLeft, name)

    def _cbGroupJoined(self, result, name):
        groupConv = self.chat.getGroupConversation(self.getGroup(name))
        groupConv.showGroupMessage('sys', 'you joined')
        self.perspective.callRemote('getGroupMembers', name)

    def _cbGroupLeft(self, result, name):
        print('left', name)
        groupConv = self.chat.getGroupConversation(self.getGroup(name), 1)
        groupConv.showGroupMessage('sys', 'you left')

    def connected(self, perspective):
        print('Connected Words Client!', perspective)
        if self._logonDeferred is not None:
            self._logonDeferred.callback(self)
        self.perspective = perspective
        self.chat.getContactsList()