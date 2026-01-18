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
def _show_log(self):
    text = '\n'.join(self._log_messages)
    ShowText(self.top, 'NLTK Downloader Log', text)