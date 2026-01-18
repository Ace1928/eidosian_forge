import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
class GetEnvironmentDictionary(UtilityProcessProtocol):
    """
    Protocol which will read a serialized environment dict from a process
    and expose it to interested parties.
    """
    programName = b'twisted.test.process_getenv'

    def parseChunks(self, chunks):
        """
        Parse the output from the process to which this protocol was
        connected, which is a single unterminated line of \\0-separated
        strings giving key value pairs of the environment from that process.
        Return this as a dictionary.
        """
        environBytes = b''.join(chunks)
        if not environBytes:
            return {}
        environb = iter(environBytes.split(b'\x00'))
        d = {}
        while 1:
            try:
                k = next(environb)
            except StopIteration:
                break
            else:
                v = next(environb)
                d[k] = v
        return d