import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
class ServerFacade(object):

    def __init__(self, notifications_queue):
        self.notifications_queue = notifications_queue

    def notifyTestsCollected(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyTestsCollected', args))

    def notifyConnected(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyConnected', args))

    def notifyTestRunFinished(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyTestRunFinished', args))

    def notifyStartTest(self, *args):
        self.notifications_queue.put_nowait(ParallelNotification('notifyStartTest', args))

    def notifyTest(self, *args):
        new_args = []
        for arg in args:
            new_args.append(_encode_if_needed(arg))
        args = tuple(new_args)
        self.notifications_queue.put_nowait(ParallelNotification('notifyTest', args))