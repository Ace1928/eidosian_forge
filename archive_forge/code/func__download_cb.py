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
def _download_cb(self, download_iter, ids):
    try:
        msg = next(download_iter)
    except StopIteration:
        self._update_table_status()
        afterid = self.top.after(10, self._show_progress, 0)
        self._afterid['_download_cb'] = afterid
        return

    def show(s):
        self._progresslabel['text'] = s
        self._log(s)
    if isinstance(msg, ProgressMessage):
        self._show_progress(msg.progress)
    elif isinstance(msg, ErrorMessage):
        show(msg.message)
        if msg.package is not None:
            self._select(msg.package.id)
        self._show_progress(None)
        return
    elif isinstance(msg, StartCollectionMessage):
        show('Downloading collection %s' % msg.collection.id)
        self._log_indent += 1
    elif isinstance(msg, StartPackageMessage):
        show('Downloading package %s' % msg.package.id)
    elif isinstance(msg, UpToDateMessage):
        show('Package %s is up-to-date!' % msg.package.id)
    elif isinstance(msg, FinishDownloadMessage):
        show('Finished downloading %r.' % msg.package.id)
    elif isinstance(msg, StartUnzipMessage):
        show('Unzipping %s' % msg.package.filename)
    elif isinstance(msg, FinishCollectionMessage):
        self._log_indent -= 1
        show('Finished downloading collection %r.' % msg.collection.id)
        self._clear_mark(msg.collection.id)
    elif isinstance(msg, FinishPackageMessage):
        self._clear_mark(msg.package.id)
    afterid = self.top.after(self._DL_DELAY, self._download_cb, download_iter, ids)
    self._afterid['_download_cb'] = afterid