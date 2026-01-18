import os
import re
import struct
from unittest import skipIf
from hamcrest import assert_that, equal_to
from twisted.internet import defer
from twisted.internet.error import ConnectionLost
from twisted.internet.testing import StringTransport
from twisted.protocols import loopback
from twisted.python import components
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
def _runAsUser(self, f, *args, **kw):
    try:
        f = iter(f)
    except TypeError:
        f = [(f, args, kw)]
    for i in f:
        func = i[0]
        args = len(i) > 1 and i[1] or ()
        kw = len(i) > 2 and i[2] or {}
        r = func(*args, **kw)
    return r