import sys
import threading
import traceback
import warnings
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle.pydev_imports import xmlrpclib, _queue
from _pydevd_bundle.pydevd_constants import Null
class ParallelNotification(object):

    def __init__(self, method, args):
        self.method = method
        self.args = args

    def to_tuple(self):
        return (self.method, self.args)