from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class EntityReferenceTests(TestCase):
    """
    Tests for L{EntityReference}.
    """

    def test_isEqualToNode(self) -> None:
        """
        L{EntityReference.isEqualToNode} returns C{True} if and only if passed
        a L{EntityReference} with the same C{eref}.
        """
        self.assertTrue(microdom.EntityReference('quot').isEqualToNode(microdom.EntityReference('quot')))
        self.assertFalse(microdom.EntityReference('quot').isEqualToNode(microdom.EntityReference('apos')))