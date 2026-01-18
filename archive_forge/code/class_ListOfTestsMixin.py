import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class ListOfTestsMixin:
    """
    Base class for testing L{ListOf}, a parameterized zero-or-more argument
    type.

    @ivar elementType: Subclasses should set this to an L{Argument}
        instance.  The tests will make a L{ListOf} using this.

    @ivar strings: Subclasses should set this to a dictionary mapping some
        number of keys -- as BYTE strings -- to the correct serialized form
        for some example values. These should agree with what L{elementType}
        produces/accepts.

    @ivar objects: Subclasses should set this to a dictionary with the same
        keys as C{strings} -- as NATIVE strings -- and with values which are
        the lists which should serialize to the values in the C{strings}
        dictionary.
    """

    def test_toBox(self):
        """
        L{ListOf.toBox} extracts the list of objects from the C{objects}
        dictionary passed to it, using the C{name} key also passed to it,
        serializes each of the elements in that list using the L{Argument}
        instance previously passed to its initializer, combines the serialized
        results, and inserts the result into the C{strings} dictionary using
        the same C{name} key.
        """
        stringList = amp.ListOf(self.elementType)
        strings = amp.AmpBox()
        for key in self.objects:
            stringList.toBox(key.encode('ascii'), strings, self.objects.copy(), None)
        self.assertEqual(strings, self.strings)

    def test_fromBox(self):
        """
        L{ListOf.fromBox} reverses the operation performed by L{ListOf.toBox}.
        """
        stringList = amp.ListOf(self.elementType)
        objects = {}
        for key in self.strings:
            stringList.fromBox(key, self.strings.copy(), objects, None)
        self.assertEqual(objects, self.objects)