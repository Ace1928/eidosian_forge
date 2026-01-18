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
class InitCatalog(CommandMixin):
    description = 'create a new catalog based on a POT file'
    user_options = [('domain=', 'D', "domain of PO file (default 'messages')"), ('input-file=', 'i', 'name of the input file'), ('output-dir=', 'd', 'path to output directory'), ('output-file=', 'o', "name of the output file (default '<output_dir>/<locale>/LC_MESSAGES/<domain>.po')"), ('locale=', 'l', 'locale for the new localized catalog'), ('width=', 'w', 'set output line width (default 76)'), ('no-wrap', None, 'do not break long message lines, longer than the output line width, into several lines')]
    boolean_options = ['no-wrap']

    def initialize_options(self):
        self.output_dir = None
        self.output_file = None
        self.input_file = None
        self.locale = None
        self.domain = 'messages'
        self.no_wrap = False
        self.width = None

    def finalize_options(self):
        if not self.input_file:
            raise OptionError('you must specify the input file')
        if not self.locale:
            raise OptionError('you must provide a locale for the new catalog')
        try:
            self._locale = Locale.parse(self.locale)
        except UnknownLocaleError as e:
            raise OptionError(e) from e
        if not self.output_file and (not self.output_dir):
            raise OptionError('you must specify the output directory')
        if not self.output_file:
            self.output_file = os.path.join(self.output_dir, self.locale, 'LC_MESSAGES', f'{self.domain}.po')
        if not os.path.exists(os.path.dirname(self.output_file)):
            os.makedirs(os.path.dirname(self.output_file))
        if self.no_wrap and self.width:
            raise OptionError("'--no-wrap' and '--width' are mutually exclusive")
        if not self.no_wrap and (not self.width):
            self.width = 76
        elif self.width is not None:
            self.width = int(self.width)

    def run(self):
        self.log.info('creating catalog %s based on %s', self.output_file, self.input_file)
        with open(self.input_file, 'rb') as infile:
            catalog = read_po(infile, locale=self.locale)
        catalog.locale = self._locale
        catalog.revision_date = datetime.datetime.now(LOCALTZ)
        catalog.fuzzy = False
        with open(self.output_file, 'wb') as outfile:
            write_po(outfile, catalog, width=self.width)