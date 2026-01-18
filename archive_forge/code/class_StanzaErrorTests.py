from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
class StanzaErrorTests(unittest.TestCase):
    """
    Tests for L{error.StreamError}.
    """

    def test_typeRemoteServerTimeout(self) -> None:
        """
        Remote Server Timeout should yield type wait, code 504.
        """
        e = error.StanzaError('remote-server-timeout')
        self.assertEqual('wait', e.type)
        self.assertEqual('504', e.code)

    def test_getElementPlain(self) -> None:
        """
        Test getting an element for a plain stanza error.
        """
        e = error.StanzaError('feature-not-implemented')
        element = e.getElement()
        self.assertEqual(element.uri, None)
        self.assertEqual(element['type'], 'cancel')
        self.assertEqual(element['code'], '501')

    def test_getElementType(self) -> None:
        """
        Test getting an element for a stanza error with a given type.
        """
        e = error.StanzaError('feature-not-implemented', 'auth')
        element = e.getElement()
        self.assertEqual(element.uri, None)
        self.assertEqual(element['type'], 'auth')
        self.assertEqual(element['code'], '501')

    def test_getElementConditionNamespace(self) -> None:
        """
        Test that the error condition element has the correct namespace.
        """
        e = error.StanzaError('feature-not-implemented')
        element = e.getElement()
        self.assertEqual(NS_XMPP_STANZAS, getattr(element, 'feature-not-implemented').uri)

    def test_getElementTextNamespace(self) -> None:
        """
        Test that the error text element has the correct namespace.
        """
        e = error.StanzaError('feature-not-implemented', text='text')
        element = e.getElement()
        self.assertEqual(NS_XMPP_STANZAS, element.text.uri)

    def test_toResponse(self) -> None:
        """
        Test an error response is generated from a stanza.

        The addressing on the (new) response stanza should be reversed, an
        error child (with proper properties) added and the type set to
        C{'error'}.
        """
        stanza = domish.Element(('jabber:client', 'message'))
        stanza['type'] = 'chat'
        stanza['to'] = 'user1@example.com'
        stanza['from'] = 'user2@example.com/resource'
        e = error.StanzaError('service-unavailable')
        response = e.toResponse(stanza)
        self.assertNotIdentical(response, stanza)
        self.assertEqual(response['from'], 'user1@example.com')
        self.assertEqual(response['to'], 'user2@example.com/resource')
        self.assertEqual(response['type'], 'error')
        self.assertEqual(response.error.children[0].name, 'service-unavailable')
        self.assertEqual(response.error['type'], 'cancel')
        self.assertNotEqual(stanza.children, response.children)