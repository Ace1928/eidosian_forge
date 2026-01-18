from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
def _compactRepr(obj: object, alwaysShow: Sequence[str] | None=None, flagNames: Sequence[str] | None=None, fieldNames: Sequence[str] | None=None, sectionNames: Sequence[str] | None=None) -> str:
    """
    Return a L{str} representation of C{obj} which only shows fields with
    non-default values, flags which are True and sections which have been
    explicitly set.

    @param obj: The instance whose repr is being generated.
    @param alwaysShow: A L{list} of field names which should always be shown.
    @param flagNames: A L{list} of flag attribute names which should be shown if
        they are L{True}.
    @param fieldNames: A L{list} of field attribute names which should be shown
        if they have non-default values.
    @param sectionNames: A L{list} of section attribute names which should be
        shown if they have been assigned a value.

    @return: A L{str} representation of C{obj}.
    """
    if alwaysShow is None:
        alwaysShow = []
    if flagNames is None:
        flagNames = []
    if fieldNames is None:
        fieldNames = []
    if sectionNames is None:
        sectionNames = []
    setFlags = []
    for name in flagNames:
        if name in alwaysShow or getattr(obj, name, False) == True:
            setFlags.append(name)
    displayableArgs = _getDisplayableArguments(obj, alwaysShow, fieldNames)
    out = ['<', obj.__class__.__name__] + displayableArgs
    if setFlags:
        out.append(' flags={}'.format(','.join(setFlags)))
    for name in sectionNames:
        section = getattr(obj, name, [])
        if section:
            out.append(f' {name}={section!r}')
    out.append('>')
    return ''.join(out)