from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
class PageRedirectTests(unittest.TestCase):
    """
    Tests for how L{PageRedirect} attributes are initialized.
    """

    def test_noMessageValidStatus(self) -> None:
        """
        If no C{message} argument is passed to the L{PageRedirect} constructor
        and the C{code} argument is a valid HTTP status code, C{code} is mapped
        to a descriptive string to which C{message} is assigned.
        """
        e = error.PageRedirect(b'200', location=b'/foo')
        self.assertEqual(e.message, b'OK to /foo')

    def test_noMessageValidStatusNoLocation(self) -> None:
        """
        If no C{message} argument is passed to the L{PageRedirect} constructor
        and C{location} is also empty and the C{code} argument is a valid HTTP
        status code, C{code} is mapped to a descriptive string to which
        C{message} is assigned without trying to include an empty location.
        """
        e = error.PageRedirect(b'200')
        self.assertEqual(e.message, b'OK')

    def test_noMessageInvalidStatusLocationExists(self) -> None:
        """
        If no C{message} argument is passed to the L{PageRedirect} constructor
        and C{code} isn't a valid HTTP status code, C{message} stays L{None}.
        """
        e = error.PageRedirect(b'999', location=b'/foo')
        self.assertEqual(e.message, None)

    def test_messageExistsLocationExists(self) -> None:
        """
        If a C{message} argument is passed to the L{PageRedirect} constructor,
        the C{message} isn't affected by the value of C{status}.
        """
        e = error.PageRedirect(b'200', b'My own message', location=b'/foo')
        self.assertEqual(e.message, b'My own message to /foo')

    def test_messageExistsNoLocation(self) -> None:
        """
        If a C{message} argument is passed to the L{PageRedirect} constructor
        and no location is provided, C{message} doesn't try to include the
        empty location.
        """
        e = error.PageRedirect(b'200', b'My own message')
        self.assertEqual(e.message, b'My own message')