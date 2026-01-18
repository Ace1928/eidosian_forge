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
def _simple_interactive_config(self):
    self._show_config()
    while True:
        print()
        self._simple_interactive_menu('s) Show Config', 'u) Set Server URL', 'd) Set Data Dir', 'm) Main Menu')
        user_input = input('Config> ').strip().lower()
        if user_input == 's':
            self._show_config()
        elif user_input == 'd':
            new_dl_dir = input('  New Directory> ').strip()
            if new_dl_dir in ('', 'x', 'q', 'X', 'Q'):
                print('  Cancelled!')
            elif os.path.isdir(new_dl_dir):
                self._ds.download_dir = new_dl_dir
            else:
                print('Directory %r not found!  Create it first.' % new_dl_dir)
        elif user_input == 'u':
            new_url = input('  New URL> ').strip()
            if new_url in ('', 'x', 'q', 'X', 'Q'):
                print('  Cancelled!')
            else:
                if not new_url.startswith(('http://', 'https://')):
                    new_url = 'http://' + new_url
                try:
                    self._ds.url = new_url
                except Exception as e:
                    print(f'Error reading <{new_url!r}>:\n  {e}')
        elif user_input == 'm':
            break