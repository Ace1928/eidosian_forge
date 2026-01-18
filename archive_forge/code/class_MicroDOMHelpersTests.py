from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
class MicroDOMHelpersTests(DOMHelpersTestsMixin, TestCase):
    dom = microdom

    def test_gatherTextNodesDropsWhitespace(self):
        """
        Microdom discards whitespace-only text nodes, so L{gatherTextNodes}
        returns only the text from nodes which had non-whitespace characters.
        """
        doc4_xml = '<html>\n  <head>\n  </head>\n  <body>\n    stuff\n  </body>\n</html>\n'
        doc4 = self.dom.parseString(doc4_xml)
        actual = domhelpers.gatherTextNodes(doc4)
        expected = '\n    stuff\n  '
        self.assertEqual(actual, expected)
        actual = domhelpers.gatherTextNodes(doc4.documentElement)
        self.assertEqual(actual, expected)

    def test_textEntitiesNotDecoded(self):
        """
        Microdom does not decode entities in text nodes.
        """
        doc5_xml = '<x>Souffl&amp;</x>'
        doc5 = self.dom.parseString(doc5_xml)
        actual = domhelpers.gatherTextNodes(doc5)
        expected = 'Souffl&amp;'
        self.assertEqual(actual, expected)
        actual = domhelpers.gatherTextNodes(doc5.documentElement)
        self.assertEqual(actual, expected)

    def test_deprecation(self):
        """
        An import will raise the deprecation warning.
        """
        reload(domhelpers)
        warnings = self.flushWarnings([self.test_deprecation])
        self.assertEqual(1, len(warnings))
        self.assertEqual('twisted.web.domhelpers was deprecated at Twisted 23.10.0', warnings[0]['message'])