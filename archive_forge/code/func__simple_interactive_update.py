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
def _simple_interactive_update(self):
    while True:
        stale_packages = []
        stale = partial = False
        for info in sorted(getattr(self._ds, 'packages')(), key=str):
            if self._ds.status(info) == self._ds.STALE:
                stale_packages.append((info.id, info.name))
        print()
        if stale_packages:
            print('Will update following packages (o=ok; x=cancel)')
            for pid, pname in stale_packages:
                name = textwrap.fill('-' * 27 + pname, 75, subsequent_indent=27 * ' ')[27:]
                print('  [ ] {} {}'.format(pid.ljust(20, '.'), name))
            print()
            user_input = input('  Identifier> ')
            if user_input.lower() == 'o':
                for pid, pname in stale_packages:
                    try:
                        self._ds.download(pid, prefix='    ')
                    except (OSError, ValueError) as e:
                        print(e)
                break
            elif user_input.lower() in ('x', 'q', ''):
                return
        else:
            print('Nothing to update.')
            return