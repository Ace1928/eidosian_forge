from __future__ import annotations
import datetime
import fnmatch
import logging
import optparse
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from configparser import RawConfigParser
from io import StringIO
from typing import Iterable
from babel import Locale, localedata
from babel import __version__ as VERSION
from babel.core import UnknownLocaleError
from babel.messages.catalog import DEFAULT_HEADER, Catalog
from babel.messages.extract import (
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po, write_po
from babel.util import LOCALTZ
def _build_callback(self, path: str):

    def callback(filename: str, method: str, options: dict):
        if method == 'ignore':
            return
        if os.path.isfile(path):
            filepath = path
        else:
            filepath = os.path.normpath(os.path.join(path, filename))
        optstr = ''
        if options:
            opt_values = ', '.join((f'{k}="{v}"' for k, v in options.items()))
            optstr = f' ({opt_values})'
        self.log.info('extracting messages from %s%s', filepath, optstr)
    return callback