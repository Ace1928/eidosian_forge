from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
class StubGroupConversation(GroupConversation):

    def setTopic(self, topic, nickname):
        self.topic = topic
        self.topicSetBy = nickname

    def show(self):
        pass

    def showGroupMessage(self, sender, text, metadata=None):
        self.metadata = metadata
        self.text = text
        self.metadata = metadata