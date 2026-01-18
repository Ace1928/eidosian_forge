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
def _find_collections(root):
    """
    Helper for ``build_index()``: Yield a list of ElementTree.Element
    objects, each holding the xml for a single package collection.
    """
    for dirname, _subdirs, files in os.walk(root):
        for filename in files:
            if filename.endswith('.xml'):
                xmlfile = os.path.join(dirname, filename)
                yield ElementTree.parse(xmlfile).getroot()