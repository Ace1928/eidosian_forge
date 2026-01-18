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
class SelectDownloadDirMessage(DownloaderMessage):
    """Indicates what download directory the data server is using"""

    def __init__(self, download_dir):
        self.download_dir = download_dir