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
def _color_table(self):
    for row in range(len(self._table)):
        bg, sbg = self._ROW_COLOR[self._table[row, 'Status']]
        fg, sfg = ('black', 'white')
        self._table.rowconfig(row, foreground=fg, selectforeground=sfg, background=bg, selectbackground=sbg)
        self._table.itemconfigure(row, 0, foreground=self._MARK_COLOR[0], background=self._MARK_COLOR[1])