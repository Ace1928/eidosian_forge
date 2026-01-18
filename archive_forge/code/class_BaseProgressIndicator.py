from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
class BaseProgressIndicator(object):
    """BaseProgressIndicator(name)

    A progress indicator helps display the progress of a task to the
    user. Progress can be pending, running, finished or failed.

    Each task has:
      * a name - a short description of what needs to be done.
      * an action - the current action in performing the task (e.g. a subtask)
      * progress - how far the task is completed
      * max - max number of progress units. If 0, the progress is indefinite
      * unit - the units in which the progress is counted
      * status - 0: pending, 1: in progress, 2: finished, 3: failed

    This class defines an abstract interface. Subclasses should implement
    _start, _stop, _update_progress(progressText), _write(message).
    """

    def __init__(self, name):
        self._name = name
        self._action = ''
        self._unit = ''
        self._max = 0
        self._status = 0
        self._last_progress_update = 0

    def start(self, action='', unit='', max=0):
        """start(action='', unit='', max=0)

        Start the progress. Optionally specify an action, a unit,
        and a maximum progress value.
        """
        if self._status == 1:
            self.finish()
        self._action = action
        self._unit = unit
        self._max = max
        self._progress = 0
        self._status = 1
        self._start()

    def status(self):
        """status()

        Get the status of the progress - 0: pending, 1: in progress,
        2: finished, 3: failed
        """
        return self._status

    def set_progress(self, progress=0, force=False):
        """set_progress(progress=0, force=False)

        Set the current progress. To avoid unnecessary progress updates
        this will only have a visual effect if the time since the last
        update is > 0.1 seconds, or if force is True.
        """
        self._progress = progress
        if not (force or time.time() - self._last_progress_update > 0.1):
            return
        self._last_progress_update = time.time()
        unit = self._unit or ''
        progressText = ''
        if unit == '%':
            progressText = '%2.1f%%' % progress
        elif self._max > 0:
            percent = 100 * float(progress) / self._max
            progressText = '%i/%i %s (%2.1f%%)' % (progress, self._max, unit, percent)
        elif progress > 0:
            if isinstance(progress, float):
                progressText = '%0.4g %s' % (progress, unit)
            else:
                progressText = '%i %s' % (progress, unit)
        self._update_progress(progressText)

    def increase_progress(self, extra_progress):
        """increase_progress(extra_progress)

        Increase the progress by a certain amount.
        """
        self.set_progress(self._progress + extra_progress)

    def finish(self, message=None):
        """finish(message=None)

        Finish the progress, optionally specifying a message. This will
        not set the progress to the maximum.
        """
        self.set_progress(self._progress, True)
        self._status = 2
        self._stop()
        if message is not None:
            self._write(message)

    def fail(self, message=None):
        """fail(message=None)

        Stop the progress with a failure, optionally specifying a message.
        """
        self.set_progress(self._progress, True)
        self._status = 3
        self._stop()
        message = 'FAIL ' + (message or '')
        self._write(message)

    def write(self, message):
        """write(message)

        Write a message during progress (such as a warning).
        """
        if self.__class__ == BaseProgressIndicator:
            print(message)
        else:
            return self._write(message)

    def _start(self):
        pass

    def _stop(self):
        pass

    def _update_progress(self, progressText):
        pass

    def _write(self, message):
        pass