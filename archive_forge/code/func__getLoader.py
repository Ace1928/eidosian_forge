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
def _getLoader(config: Options) -> runner.TestLoader:
    loader = runner.TestLoader()
    if config['random']:
        randomer = random.Random()
        randomer.seed(config['random'])
        loader.sorter = lambda x: randomer.random()
        print('Running tests shuffled with seed %d\n' % config['random'])
    elif config['order']:
        _, sorter = _runOrders[config['order']]
        loader.sorter = sorter
    if not config['until-failure']:
        loader.suiteFactory = runner.DestructiveTestSuite
    return loader