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
class _DownloadThread(threading.Thread):

    def __init__(self, data_server, items, lock, message_queue, abort):
        self.data_server = data_server
        self.items = items
        self.lock = lock
        self.message_queue = message_queue
        self.abort = abort
        threading.Thread.__init__(self)

    def run(self):
        for msg in self.data_server.incr_download(self.items):
            self.lock.acquire()
            self.message_queue.append(msg)
            if self.abort:
                self.message_queue.append('aborted')
                self.lock.release()
                return
            self.lock.release()
        self.lock.acquire()
        self.message_queue.append('finished')
        self.lock.release()