import signal
import time
from types import FrameType
from typing import Callable, List, Optional, Tuple, Union, cast
from twisted.internet.abstract import FileDescriptor
from twisted.internet.defer import Deferred
from twisted.internet.error import ReactorAlreadyRunning, ReactorNotRestartable
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
class OKFileDescriptor(FileDescriptor):

    def connectionLost(self, reason: Failure) -> None:
        lostOK[0] = True