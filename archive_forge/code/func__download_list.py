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
def _download_list(self, items, download_dir, force):
    for i in range(len(items)):
        try:
            items[i] = self._info_or_id(items[i])
        except (OSError, ValueError) as e:
            yield ErrorMessage(items[i], e)
            return
    num_packages = sum((self._num_packages(item) for item in items))
    progress = 0
    for i, item in enumerate(items):
        if isinstance(item, Package):
            delta = 1.0 / num_packages
        else:
            delta = len(item.packages) / num_packages
        for msg in self.incr_download(item, download_dir, force):
            if isinstance(msg, ProgressMessage):
                yield ProgressMessage(progress + msg.progress * delta)
            else:
                yield msg
        progress += 100 * delta