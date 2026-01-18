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
def _run_domain(self, domain):
    po_files = []
    mo_files = []
    if not self.input_file:
        if self.locale:
            po_files.append((self.locale, os.path.join(self.directory, self.locale, 'LC_MESSAGES', f'{domain}.po')))
            mo_files.append(os.path.join(self.directory, self.locale, 'LC_MESSAGES', f'{domain}.mo'))
        else:
            for locale in os.listdir(self.directory):
                po_file = os.path.join(self.directory, locale, 'LC_MESSAGES', f'{domain}.po')
                if os.path.exists(po_file):
                    po_files.append((locale, po_file))
                    mo_files.append(os.path.join(self.directory, locale, 'LC_MESSAGES', f'{domain}.mo'))
    else:
        po_files.append((self.locale, self.input_file))
        if self.output_file:
            mo_files.append(self.output_file)
        else:
            mo_files.append(os.path.join(self.directory, self.locale, 'LC_MESSAGES', f'{domain}.mo'))
    if not po_files:
        raise OptionError('no message catalogs found')
    catalogs_and_errors = {}
    for idx, (locale, po_file) in enumerate(po_files):
        mo_file = mo_files[idx]
        with open(po_file, 'rb') as infile:
            catalog = read_po(infile, locale)
        if self.statistics:
            translated = 0
            for message in list(catalog)[1:]:
                if message.string:
                    translated += 1
            percentage = 0
            if len(catalog):
                percentage = translated * 100 // len(catalog)
            self.log.info('%d of %d messages (%d%%) translated in %s', translated, len(catalog), percentage, po_file)
        if catalog.fuzzy and (not self.use_fuzzy):
            self.log.info('catalog %s is marked as fuzzy, skipping', po_file)
            continue
        catalogs_and_errors[catalog] = catalog_errors = list(catalog.check())
        for message, errors in catalog_errors:
            for error in errors:
                self.log.error('error: %s:%d: %s', po_file, message.lineno, error)
        self.log.info('compiling catalog %s to %s', po_file, mo_file)
        with open(mo_file, 'wb') as outfile:
            write_mo(outfile, catalog, use_fuzzy=self.use_fuzzy)
    return catalogs_and_errors