import time
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
from _pydevd_bundle.pydevd_constants import get_thread_id
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import ObjectWrapper, wrap_attr
import pydevd_file_utils
from _pydev_bundle import pydev_log
import sys
from urllib.parse import quote
class NameManager:

    def __init__(self, name_prefix):
        self.tasks = {}
        self.last = 0
        self.prefix = name_prefix

    def get(self, id):
        if id not in self.tasks:
            self.last += 1
            self.tasks[id] = self.prefix + '-' + str(self.last)
        return self.tasks[id]