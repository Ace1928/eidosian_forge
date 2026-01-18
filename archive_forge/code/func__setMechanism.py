from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def _setMechanism(self, name):
    """
        Set up the XML Stream to have a SASL feature with the given mechanism.
        """
    feature = domish.Element((NS_XMPP_SASL, 'mechanisms'))
    feature.addElement('mechanism', content=name)
    self.xmlstream.features[feature.uri, feature.name] = feature
    self.init.setMechanism()
    return self.init.mechanism.name