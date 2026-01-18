import errno
import logging
import logging.config
import logging.handlers
import os
import pyinotify
import stat
import time
class _FileKeeper(pyinotify.ProcessEvent):

    def my_init(self, watched_handler, watched_file):
        self._watched_handler = watched_handler
        self._watched_file = watched_file

    def process_default(self, event):
        if event.name == self._watched_file:
            self._watched_handler.reopen_file()