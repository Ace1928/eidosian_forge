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
def _init_progressbar(self):
    c = self._progressbar
    width, height = (int(c['width']), int(c['height']))
    for i in range(0, int(c['width']) * 2 // self._gradient_width):
        c.create_line(i * self._gradient_width + 20, -20, i * self._gradient_width - height - 20, height + 20, width=self._gradient_width, fill='#%02x0000' % (80 + abs(i % 6 - 3) * 12))
    c.addtag_all('gradient')
    c.itemconfig('gradient', state='hidden')
    c.addtag_withtag('redbox', c.create_rectangle(0, 0, 0, 0, fill=self._PROGRESS_COLOR[0]))