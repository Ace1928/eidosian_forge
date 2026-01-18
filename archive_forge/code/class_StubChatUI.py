from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
class StubChatUI(ChatUI):

    def getConversation(self, group, Class=StubConversation, stayHidden=0):
        return ChatUI.getGroupConversation(self, group, Class, stayHidden)

    def getGroupConversation(self, group, Class=StubGroupConversation, stayHidden=0):
        return ChatUI.getGroupConversation(self, group, Class, stayHidden)