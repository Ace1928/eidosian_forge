from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
def _sendTopic(self, group):
    """
        Send the topic of the given group to this user, if it has one.
        """
    topic = group.meta.get('topic')
    if topic:
        author = group.meta.get('topic_author') or '<noone>'
        date = group.meta.get('topic_date', 0)
        self.topic(self.name, '#' + group.name, topic)
        self.topicAuthor(self.name, '#' + group.name, author, date)