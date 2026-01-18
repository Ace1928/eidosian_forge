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
def _launchWorkerProcesses(self, spawner, protocols, arguments):
    """
        Spawn processes from a list of process protocols.

        @param spawner: A C{IReactorProcess.spawnProcess} implementation.

        @param protocols: An iterable of C{ProcessProtocol} instances.

        @param arguments: Extra arguments passed to the processes.
        """
    workertrialPath = theSystemPath['twisted.trial._dist.workertrial'].filePath.path
    childFDs = {0: 'w', 1: 'r', 2: 'r', _WORKER_AMP_STDIN: 'w', _WORKER_AMP_STDOUT: 'r'}
    environ = os.environ.copy()
    environ['PYTHONPATH'] = os.pathsep.join(sys.path)
    for worker in protocols:
        args = [sys.executable, workertrialPath]
        args.extend(arguments)
        spawner(worker, sys.executable, args=args, childFDs=childFDs, env=environ)