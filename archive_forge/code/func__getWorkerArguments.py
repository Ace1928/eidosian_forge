import gc
import inspect
import os
import pdb
import random
import sys
import time
import trace
import warnings
from typing import NoReturn, Optional, Type
from twisted import plugin
from twisted.application import app
from twisted.internet import defer
from twisted.python import failure, reflect, usage
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedModule
from twisted.trial import itrial, runner
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial.unittest import TestSuite
def _getWorkerArguments(self):
    """
        Return a list of options to pass to distributed workers.
        """
    args = []
    for option in self._workerFlags:
        if self.get(option) is not None:
            if self[option]:
                args.append(f'--{option}')
    for option in self._workerParameters:
        if self.get(option) is not None:
            args.extend([f'--{option}', str(self[option])])
    return args