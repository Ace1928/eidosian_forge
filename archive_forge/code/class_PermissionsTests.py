from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
class PermissionsTests(BytesTestCase):
    """
    Test Permissions and RWX classes
    """

    def assertNotUnequal(self, first: T, second: object, msg: Optional[str]=None) -> T:
        """
        Tests that C{first} != C{second} is false.  This method tests the
        __ne__ method, as opposed to L{assertEqual} (C{first} == C{second}),
        which tests the __eq__ method.

        Note: this should really be part of trial
        """
        if first != second:
            if msg is None:
                msg = ''
            if len(msg) > 0:
                msg += '\n'
            raise self.failureException('%snot not unequal (__ne__ not implemented correctly):\na = %s\nb = %s\n' % (msg, pformat(first), pformat(second)))
        return first

    def test_test(self) -> None:
        """
        Self-test for assertNotUnequal to make sure the assertion works.
        """
        with self.assertRaises(AssertionError) as ae:
            self.assertNotUnequal(3, 4, 'custom message')
        self.assertIn('__ne__ not implemented correctly', str(ae.exception))
        self.assertIn('custom message', str(ae.exception))
        with self.assertRaises(AssertionError) as ae2:
            self.assertNotUnequal(4, 3)
        self.assertIn('__ne__ not implemented correctly', str(ae2.exception))
        self.assertNotUnequal(3, 3)

    def test_rwxFromBools(self) -> None:
        """
        L{RWX}'s constructor takes a set of booleans
        """
        for r in (True, False):
            for w in (True, False):
                for x in (True, False):
                    rwx = filepath.RWX(r, w, x)
                    self.assertEqual(rwx.read, r)
                    self.assertEqual(rwx.write, w)
                    self.assertEqual(rwx.execute, x)
        rwx = filepath.RWX(True, True, True)
        self.assertTrue(rwx.read and rwx.write and rwx.execute)

    def test_rwxEqNe(self) -> None:
        """
        L{RWX}'s created with the same booleans are equivalent.  If booleans
        are different, they are not equal.
        """
        for r in (True, False):
            for w in (True, False):
                for x in (True, False):
                    self.assertEqual(filepath.RWX(r, w, x), filepath.RWX(r, w, x))
                    self.assertNotUnequal(filepath.RWX(r, w, x), filepath.RWX(r, w, x))
        self.assertNotEqual(filepath.RWX(True, True, True), filepath.RWX(True, True, False))
        self.assertNotEqual(3, filepath.RWX(True, True, True))

    def test_rwxShorthand(self) -> None:
        """
        L{RWX}'s shorthand string should be 'rwx' if read, write, and execute
        permission bits are true.  If any of those permissions bits are false,
        the character is replaced by a '-'.
        """

        def getChar(val: bool, letter: str) -> str:
            if val:
                return letter
            return '-'
        for r in (True, False):
            for w in (True, False):
                for x in (True, False):
                    rwx = filepath.RWX(r, w, x)
                    self.assertEqual(rwx.shorthand(), getChar(r, 'r') + getChar(w, 'w') + getChar(x, 'x'))
        self.assertEqual(filepath.RWX(True, False, True).shorthand(), 'r-x')

    def test_permissionsFromStat(self) -> None:
        """
        L{Permissions}'s constructor takes a valid permissions bitmask and
        parsaes it to produce the correct set of boolean permissions.
        """

        def _rwxFromStat(statModeInt: int, who: str) -> filepath.RWX:

            def getPermissionBit(what: str, who: str) -> bool:
                constant: int = getattr(stat, f'S_I{what}{who}')
                return statModeInt & constant > 0
            return filepath.RWX(*(getPermissionBit(what, who) for what in ('R', 'W', 'X')))
        for u in range(0, 8):
            for g in range(0, 8):
                for o in range(0, 8):
                    chmodString = '%d%d%d' % (u, g, o)
                    chmodVal = int(chmodString, 8)
                    perm = filepath.Permissions(chmodVal)
                    self.assertEqual(perm.user, _rwxFromStat(chmodVal, 'USR'), f'{chmodString}: got user: {perm.user}')
                    self.assertEqual(perm.group, _rwxFromStat(chmodVal, 'GRP'), f'{chmodString}: got group: {perm.group}')
                    self.assertEqual(perm.other, _rwxFromStat(chmodVal, 'OTH'), f'{chmodString}: got other: {perm.other}')
        perm = filepath.Permissions(511)
        for who in ('user', 'group', 'other'):
            for what in ('read', 'write', 'execute'):
                self.assertTrue(getattr(getattr(perm, who), what))

    def test_permissionsEq(self) -> None:
        """
        Two L{Permissions}'s that are created with the same bitmask
        are equivalent
        """
        self.assertEqual(filepath.Permissions(511), filepath.Permissions(511))
        self.assertNotUnequal(filepath.Permissions(511), filepath.Permissions(511))
        self.assertNotEqual(filepath.Permissions(511), filepath.Permissions(448))
        self.assertNotEqual(3, filepath.Permissions(511))

    def test_permissionsShorthand(self) -> None:
        """
        L{Permissions}'s shorthand string is the RWX shorthand string for its
        user permission bits, group permission bits, and other permission bits
        concatenated together, without a space.
        """
        for u in range(0, 8):
            for g in range(0, 8):
                for o in range(0, 8):
                    perm = filepath.Permissions(int('0o%d%d%d' % (u, g, o), 8))
                    self.assertEqual(perm.shorthand(), ''.join((x.shorthand() for x in (perm.user, perm.group, perm.other))))
        self.assertEqual(filepath.Permissions(504).shorthand(), 'rwxrwx---')