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
def _interactive_download(self):
    if TKINTER:
        try:
            DownloaderGUI(self).mainloop()
        except TclError:
            DownloaderShell(self).run()
    else:
        DownloaderShell(self).run()