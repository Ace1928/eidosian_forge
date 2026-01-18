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
class DownloaderShell:

    def __init__(self, dataserver):
        self._ds = dataserver

    def _simple_interactive_menu(self, *options):
        print('-' * 75)
        spc = (68 - sum((len(o) for o in options))) // (len(options) - 1) * ' '
        print('    ' + spc.join(options))
        print('-' * 75)

    def run(self):
        print('NLTK Downloader')
        while True:
            self._simple_interactive_menu('d) Download', 'l) List', ' u) Update', 'c) Config', 'h) Help', 'q) Quit')
            user_input = input('Downloader> ').strip()
            if not user_input:
                print()
                continue
            command = user_input.lower().split()[0]
            args = user_input.split()[1:]
            try:
                if command == 'l':
                    print()
                    self._ds.list(self._ds.download_dir, header=False, more_prompt=True)
                elif command == 'h':
                    self._simple_interactive_help()
                elif command == 'c':
                    self._simple_interactive_config()
                elif command in ('q', 'x'):
                    return
                elif command == 'd':
                    self._simple_interactive_download(args)
                elif command == 'u':
                    self._simple_interactive_update()
                else:
                    print('Command %r unrecognized' % user_input)
            except HTTPError as e:
                print('Error reading from server: %s' % e)
            except URLError as e:
                print('Error connecting to server: %s' % e.reason)
            print()

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

    def _simple_interactive_help(self):
        print()
        print('Commands:')
        print('  d) Download a package or collection     u) Update out of date packages')
        print('  l) List packages & collections          h) Help')
        print('  c) View & Modify Configuration          q) Quit')

    def _show_config(self):
        print()
        print('Data Server:')
        print('  - URL: <%s>' % self._ds.url)
        print('  - %d Package Collections Available' % len(self._ds.collections()))
        print('  - %d Individual Packages Available' % len(self._ds.packages()))
        print()
        print('Local Machine:')
        print('  - Data directory: %s' % self._ds.download_dir)

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