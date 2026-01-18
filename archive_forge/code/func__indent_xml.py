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
def _indent_xml(xml, prefix=''):
    """
    Helper for ``build_index()``: Given an XML ``ElementTree``, modify it
    (and its descendents) ``text`` and ``tail`` attributes to generate
    an indented tree, where each nested element is indented by 2
    spaces with respect to its parent.
    """
    if len(xml) > 0:
        xml.text = (xml.text or '').strip() + '\n' + prefix + '  '
        for child in xml:
            _indent_xml(child, prefix + '  ')
        for child in xml[:-1]:
            child.tail = (child.tail or '').strip() + '\n' + prefix + '  '
        xml[-1].tail = (xml[-1].tail or '').strip() + '\n' + prefix