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
def _show_info(self):
    print('showing info', self._ds.url)
    for entry, cb in self._info.values():
        entry['state'] = 'normal'
        entry.delete(0, 'end')
    self._info['url'][0].insert(0, self._ds.url)
    self._info['download_dir'][0].insert(0, self._ds.download_dir)
    for entry, cb in self._info.values():
        entry['state'] = 'disabled'