import os
import sys
from functools import partial
from os.path import isabs
from typing import (
from unittest import TestCase, TestSuite
from attrs import define, field, frozen
from attrs.converters import default_if_none
from twisted.internet.defer import Deferred, DeferredList, gatherResults
from twisted.internet.interfaces import IReactorCore, IReactorProcess
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
from twisted.python.modules import theSystemPath
from .._asyncrunner import _iterateTests
from ..itrial import IReporter, ITestCase
from ..reporter import UncleanWarningsReporterWrapper
from ..runner import TestHolder
from ..util import _unusedTestDirectory, openTestLog
from . import _WORKER_AMP_STDIN, _WORKER_AMP_STDOUT
from .distreporter import DistReporter
from .functional import countingCalls, discardResult, iterateWhile, takeWhile
from .worker import LocalWorker, LocalWorkerAMP, WorkerAction
def _createLocalWorkers(self, protocols: Iterable[LocalWorkerAMP], workingDirectory: FilePath[Any], logFile: TextIO) -> List[LocalWorker]:
    """
        Create local worker protocol instances and return them.

        @param protocols: The process/protocol adapters to use for the created
        workers.

        @param workingDirectory: The base path in which we should run the
            workers.

        @param logFile: The test log, for workers to write to.

        @return: A list of C{quantity} C{LocalWorker} instances.
        """
    return [LocalWorker(protocol, workingDirectory.child(str(x)), logFile) for x, protocol in enumerate(protocols)]