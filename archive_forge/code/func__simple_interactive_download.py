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
def _simple_interactive_download(self, args):
    if args:
        for arg in args:
            try:
                self._ds.download(arg, prefix='    ')
            except (OSError, ValueError) as e:
                print(e)
    else:
        while True:
            print()
            print('Download which package (l=list; x=cancel)?')
            user_input = input('  Identifier> ')
            if user_input.lower() == 'l':
                self._ds.list(self._ds.download_dir, header=False, more_prompt=True, skip_installed=True)
                continue
            elif user_input.lower() in ('x', 'q', ''):
                return
            elif user_input:
                for id in user_input.split():
                    try:
                        self._ds.download(id, prefix='    ')
                    except (OSError, ValueError) as e:
                        print(e)
                break