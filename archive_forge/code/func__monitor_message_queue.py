import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import nltk
def _monitor_message_queue(self):

    def show(s):
        self._progresslabel['text'] = s
        self._log(s)
    if not self._download_lock.acquire():
        return
    for msg in self._download_msg_queue:
        if msg == 'finished' or msg == 'aborted':
            self._update_table_status()
            self._downloading = False
            self._download_button['text'] = 'Download'
            del self._download_msg_queue[:]
            del self._download_abort_queue[:]
            self._download_lock.release()
            if msg == 'aborted':
                show('Download aborted!')
                self._show_progress(None)
            else:
                afterid = self.top.after(100, self._show_progress, None)
                self._afterid['_monitor_message_queue'] = afterid
            return
        elif isinstance(msg, ProgressMessage):
            self._show_progress(msg.progress)
        elif isinstance(msg, ErrorMessage):
            show(msg.message)
            if msg.package is not None:
                self._select(msg.package.id)
            self._show_progress(None)
            self._downloading = False
            return
        elif isinstance(msg, StartCollectionMessage):
            show('Downloading collection %r' % msg.collection.id)
            self._log_indent += 1
        elif isinstance(msg, StartPackageMessage):
            self._ds.clear_status_cache(msg.package.id)
            show('Downloading package %r' % msg.package.id)
        elif isinstance(msg, UpToDateMessage):
            show('Package %s is up-to-date!' % msg.package.id)
        elif isinstance(msg, FinishDownloadMessage):
            show('Finished downloading %r.' % msg.package.id)
        elif isinstance(msg, StartUnzipMessage):
            show('Unzipping %s' % msg.package.filename)
        elif isinstance(msg, FinishUnzipMessage):
            show('Finished installing %s' % msg.package.id)
        elif isinstance(msg, FinishCollectionMessage):
            self._log_indent -= 1
            show('Finished downloading collection %r.' % msg.collection.id)
            self._clear_mark(msg.collection.id)
        elif isinstance(msg, FinishPackageMessage):
            self._update_table_status()
            self._clear_mark(msg.package.id)
    if self._download_abort_queue:
        self._progresslabel['text'] = 'Aborting download...'
    del self._download_msg_queue[:]
    self._download_lock.release()
    afterid = self.top.after(self._MONITOR_QUEUE_DELAY, self._monitor_message_queue)
    self._afterid['_monitor_message_queue'] = afterid