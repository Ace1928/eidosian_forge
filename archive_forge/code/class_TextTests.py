from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class TextTests(TestCase):
    """
    Tests for L{Text}.
    """

    def test_isEqualToNode(self) -> None:
        """
        L{Text.isEqualToNode} returns C{True} if and only if passed a L{Text}
        which represents the same data.
        """
        self.assertTrue(microdom.Text('foo', raw=True).isEqualToNode(microdom.Text('foo', raw=True)))
        self.assertFalse(microdom.Text('foo', raw=True).isEqualToNode(microdom.Text('foo', raw=False)))
        self.assertFalse(microdom.Text('foo', raw=True).isEqualToNode(microdom.Text('bar', raw=True)))