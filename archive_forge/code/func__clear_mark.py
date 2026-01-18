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
def _clear_mark(self, id):
    for row in range(len(self._table)):
        if self._table[row, 'Identifier'] == id:
            self._table[row, 0] = ''