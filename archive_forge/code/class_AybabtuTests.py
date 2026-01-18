from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class AybabtuTests(TestCase):
    """
    L{styles._aybabtu} gets all of classes in the inheritance hierarchy of its
    argument that are strictly between L{Versioned} and the class itself.
    """

    def test_aybabtuStrictEmpty(self) -> None:
        """
        L{styles._aybabtu} of L{Versioned} itself is an empty list.
        """
        self.assertEqual(styles._aybabtu(styles.Versioned), [])

    def test_aybabtuStrictSubclass(self) -> None:
        """
        There are no classes I{between} L{VersionedSubClass} and L{Versioned},
        so L{styles._aybabtu} returns an empty list.
        """
        self.assertEqual(styles._aybabtu(VersionedSubClass), [])

    def test_aybabtuSubsubclass(self) -> None:
        """
        With a sub-sub-class of L{Versioned}, L{styles._aybabtu} returns a list
        containing the intervening subclass.
        """
        self.assertEqual(styles._aybabtu(VersionedSubSubClass), [VersionedSubClass])

    def test_aybabtuStrict(self) -> None:
        """
        For a diamond-shaped inheritance graph, L{styles._aybabtu} returns a
        list containing I{both} intermediate subclasses.
        """
        self.assertEqual(styles._aybabtu(VersionedDiamondSubClass), [VersionedSubSubClass, VersionedSubClass, SecondVersionedSubClass])