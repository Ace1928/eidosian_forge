import errno
import gc
import io
import os
import socket
from functools import wraps
from typing import Callable, ClassVar, List, Mapping, Optional, Sequence, Type
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass, verifyObject
import attr
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import (
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import (
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.test.test_tcp import (
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
class _IExhaustsFileDescriptors(Interface):
    """
    A way to trigger C{EMFILE}.
    """

    def exhaust():
        """
        Open file descriptors until C{EMFILE} is reached.

        This can raise any exception except an L{OSError} whose
        C{errno} is C{EMFILE}.  Any exception raised to the caller
        implies L{release}.
        """

    def release():
        """
        Release all file descriptors opened by L{exhaust}.
        """

    def count():
        """
        Return the number of opened file descriptors.

        @return: The number of opened file descriptors; this will be
            zero if this instance has not opened any.
        @rtype: L{int}
        """