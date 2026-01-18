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
def _check_package(pkg_xml, zipfilename, zf):
    """
    Helper for ``build_index()``: Perform some checks to make sure that
    the given package is consistent.
    """
    uid = os.path.splitext(os.path.split(zipfilename)[1])[0]
    if pkg_xml.get('id') != uid:
        raise ValueError('package identifier mismatch ({} vs {})'.format(pkg_xml.get('id'), uid))
    if sum((name != uid and (not name.startswith(uid + '/')) for name in zf.namelist())):
        raise ValueError('Zipfile %s.zip does not expand to a single subdirectory %s/' % (uid, uid))