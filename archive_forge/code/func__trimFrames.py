from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
def _trimFrames(self, frames):
    """
        Trim frames to remove internal paths.

        When a C{SynchronousTestCase} method fails synchronously, the stack
        looks like this:
         - [0]: C{SynchronousTestCase._run}
         - [1]: C{util.runWithWarningsSuppressed}
         - [2:-2]: code in the test method which failed
         - [-1]: C{_synctest.fail}

        When a C{TestCase} method fails synchronously, the stack looks like
        this:
         - [0]: C{defer.maybeDeferred}
         - [1]: C{utils.runWithWarningsSuppressed}
         - [2]: C{utils.runWithWarningsSuppressed}
         - [3:-2]: code in the test method which failed
         - [-1]: C{_synctest.fail}

        When a method fails inside a C{Deferred} (i.e., when the test method
        returns a C{Deferred}, and that C{Deferred}'s errback fires), the stack
        captured inside the resulting C{Failure} looks like this:
         - [0]: C{defer.Deferred._runCallbacks}
         - [1:-2]: code in the testmethod which failed
         - [-1]: C{_synctest.fail}

        As a result, we want to trim either [maybeDeferred, runWWS, runWWS] or
        [Deferred._runCallbacks] or [SynchronousTestCase._run, runWWS] from the
        front, and trim the [unittest.fail] from the end.

        There is also another case, when the test method is badly defined and
        contains extra arguments.

        If it doesn't recognize one of these cases, it just returns the
        original frames.

        @param frames: The C{list} of frames from the test failure.

        @return: The C{list} of frames to display.
        """
    newFrames = list(frames)
    if len(frames) < 2:
        return newFrames
    firstMethod = newFrames[0][0]
    firstFile = os.path.splitext(os.path.basename(newFrames[0][1]))[0]
    secondMethod = newFrames[1][0]
    secondFile = os.path.splitext(os.path.basename(newFrames[1][1]))[0]
    syncCase = (('_run', '_synctest'), ('runWithWarningsSuppressed', 'util'))
    asyncCase = (('maybeDeferred', 'defer'), ('runWithWarningsSuppressed', 'utils'))
    twoFrames = ((firstMethod, firstFile), (secondMethod, secondFile))
    for frame in newFrames:
        frameFile = os.path.splitext(os.path.basename(frame[1]))[0]
        if frameFile == 'compat' and frame[0] == 'reraise':
            newFrames.pop(newFrames.index(frame))
    if twoFrames == syncCase:
        newFrames = newFrames[2:]
    elif twoFrames == asyncCase:
        newFrames = newFrames[3:]
    elif (firstMethod, firstFile) == ('_runCallbacks', 'defer'):
        newFrames = newFrames[1:]
    if not newFrames:
        return newFrames
    last = newFrames[-1]
    if last[0].startswith('fail') and os.path.splitext(os.path.basename(last[1]))[0] == '_synctest':
        newFrames = newFrames[:-1]
    return newFrames