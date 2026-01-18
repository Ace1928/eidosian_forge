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
def _progress_alive(self):
    c = self._progressbar
    if not self._downloading:
        c.itemconfig('gradient', state='hidden')
    else:
        c.itemconfig('gradient', state='normal')
        x1, y1, x2, y2 = c.bbox('gradient')
        if x1 <= -100:
            c.move('gradient', self._gradient_width * 6 - 4, 0)
        else:
            c.move('gradient', -4, 0)
        afterid = self.top.after(200, self._progress_alive)
        self._afterid['_progress_alive'] = afterid