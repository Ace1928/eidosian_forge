import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _testFeatureDefault(self, name, features=None):
    """
        Features known to have default values are reported as being present by
        L{irc.ServerSupportedFeatures.hasFeature}, and their value defaults
        correctly, when they don't appear in an ISUPPORT message.
        """
    default = irc.ServerSupportedFeatures()._features[name]
    if features is None:
        features = [('DEFINITELY_NOT', 'a_feature')]
    supported = self._parse(features)
    self.assertTrue(supported.hasFeature(name))
    self.assertEqual(supported.getFeature(name), default)