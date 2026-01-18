import os
import pickle
from twisted.internet.address import UNIXAddress
from twisted.mail import smtp
from twisted.python import log
class RelayerMixin:

    def loadMessages(self, messagePaths):
        self.messages = []
        self.names = []
        for message in messagePaths:
            with open(message + '-H', 'rb') as fp:
                messageContents = pickle.load(fp)
            fp = open(message + '-D')
            messageContents.append(fp)
            self.messages.append(messageContents)
            self.names.append(message)

    def getMailFrom(self):
        if not self.messages:
            return None
        return self.messages[0][0]

    def getMailTo(self):
        if not self.messages:
            return None
        return [self.messages[0][1]]

    def getMailData(self):
        if not self.messages:
            return None
        return self.messages[0][2]

    def sentMail(self, code, resp, numOk, addresses, log):
        """Since we only use one recipient per envelope, this
        will be called with 0 or 1 addresses. We probably want
        to do something with the error message if we failed.
        """
        if code in smtp.SUCCESS:
            os.remove(self.names[0] + '-D')
            os.remove(self.names[0] + '-H')
        del self.messages[0]
        del self.names[0]