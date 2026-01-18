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
def _table_mark(self, *e):
    selection = self._table.selected_row()
    if selection >= 0:
        if self._table[selection][0] != '':
            self._table[selection, 0] = ''
        else:
            self._table[selection, 0] = 'X'
    self._table.select(delta=1)