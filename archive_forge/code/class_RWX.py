from __future__ import annotations
import base64
import errno
import os
import sys
from os import listdir, stat, utime
from os.path import (
from stat import (
from typing import (
from zope.interface import Attribute, Interface, implementer
from typing_extensions import Literal
from twisted.python.compat import cmp, comparable
from twisted.python.runtime import platform
from twisted.python.util import FancyEqMixin
from twisted.python.win32 import (
class RWX(FancyEqMixin):
    """
    A class representing read/write/execute permissions for a single user
    category (i.e. user/owner, group, or other/world).  Instantiate with
    three boolean values: readable? writable? executable?.

    @type read: C{bool}
    @ivar read: Whether permission to read is given

    @type write: C{bool}
    @ivar write: Whether permission to write is given

    @type execute: C{bool}
    @ivar execute: Whether permission to execute is given

    @since: 11.1
    """
    compareAttributes = ('read', 'write', 'execute')

    def __init__(self, readable: bool, writable: bool, executable: bool) -> None:
        self.read = readable
        self.write = writable
        self.execute = executable

    def __repr__(self) -> str:
        return 'RWX(read={}, write={}, execute={})'.format(self.read, self.write, self.execute)

    def shorthand(self) -> str:
        """
        Returns a short string representing the permission bits.  Looks like
        part of what is printed by command line utilities such as 'ls -l'
        (e.g. 'rwx')

        @return: The shorthand string.
        @rtype: L{str}
        """
        returnval = ['r', 'w', 'x']
        i = 0
        for val in (self.read, self.write, self.execute):
            if not val:
                returnval[i] = '-'
            i += 1
        return ''.join(returnval)