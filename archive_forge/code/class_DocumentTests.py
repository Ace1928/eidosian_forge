from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class DocumentTests(TestCase):
    """
    Tests for L{Document}.
    """
    doctype = 'foo PUBLIC "baz" "http://www.example.com/example.dtd"'

    def test_isEqualToNode(self) -> None:
        """
        L{Document.isEqualToNode} returns C{True} if and only if passed a
        L{Document} with the same C{doctype} and C{documentElement}.
        """
        document = microdom.Document()
        self.assertTrue(document.isEqualToNode(document))
        another = microdom.Document()
        self.assertTrue(document.isEqualToNode(another))
        document.doctype = self.doctype
        self.assertFalse(document.isEqualToNode(another))
        another.doctype = self.doctype
        self.assertTrue(document.isEqualToNode(another))
        document.appendChild(microdom.Node(object()))
        self.assertFalse(document.isEqualToNode(another))
        another.appendChild(microdom.Node(object()))
        self.assertTrue(document.isEqualToNode(another))
        document.documentElement.appendChild(microdom.Node(object()))
        self.assertFalse(document.isEqualToNode(another))

    def test_childRestriction(self) -> None:
        """
        L{Document.appendChild} raises L{ValueError} if the document already
        has a child.
        """
        document = microdom.Document()
        child = microdom.Node()
        another = microdom.Node()
        document.appendChild(child)
        self.assertRaises(ValueError, document.appendChild, another)