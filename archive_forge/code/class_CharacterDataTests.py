from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class CharacterDataTests(TestCase):
    """
    Tests for L{CharacterData}.
    """

    def test_isEqualToNode(self) -> None:
        """
        L{CharacterData.isEqualToNode} returns C{True} if and only if passed a
        L{CharacterData} with the same value.
        """
        self.assertTrue(microdom.CharacterData('foo').isEqualToNode(microdom.CharacterData('foo')))
        self.assertFalse(microdom.CharacterData('foo').isEqualToNode(microdom.CharacterData('bar')))