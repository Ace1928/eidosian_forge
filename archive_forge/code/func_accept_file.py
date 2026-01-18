import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir
import time
@accept_file.setter
def accept_file(self, accept_file):
    self._accept_file = accept_file
    for path_watcher in self._path_watchers:
        path_watcher.accept_file = accept_file