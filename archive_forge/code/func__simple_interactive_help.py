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
def _simple_interactive_help(self):
    print()
    print('Commands:')
    print('  d) Download a package or collection     u) Update out of date packages')
    print('  l) List packages & collections          h) Help')
    print('  c) View & Modify Configuration          q) Quit')