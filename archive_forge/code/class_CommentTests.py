from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class CommentTests(TestCase):
    """
    Tests for L{Comment}.
    """

    def test_isEqualToNode(self) -> None:
        """
        L{Comment.isEqualToNode} returns C{True} if and only if passed a
        L{Comment} with the same value.
        """
        self.assertTrue(microdom.Comment('foo').isEqualToNode(microdom.Comment('foo')))
        self.assertFalse(microdom.Comment('foo').isEqualToNode(microdom.Comment('bar')))