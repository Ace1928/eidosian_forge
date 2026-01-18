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
class UpdateCatalog(CommandMixin):
    description = 'update message catalogs from a POT file'
    user_options = [('domain=', 'D', "domain of PO file (default 'messages')"), ('input-file=', 'i', 'name of the input file'), ('output-dir=', 'd', 'path to base directory containing the catalogs'), ('output-file=', 'o', "name of the output file (default '<output_dir>/<locale>/LC_MESSAGES/<domain>.po')"), ('omit-header', None, 'do not include msgid  entry in header'), ('locale=', 'l', 'locale of the catalog to compile'), ('width=', 'w', 'set output line width (default 76)'), ('no-wrap', None, 'do not break long message lines, longer than the output line width, into several lines'), ('ignore-obsolete=', None, 'whether to omit obsolete messages from the output'), ('init-missing=', None, 'if any output files are missing, initialize them first'), ('no-fuzzy-matching', 'N', 'do not use fuzzy matching'), ('update-header-comment', None, 'update target header comment'), ('previous', None, 'keep previous msgids of translated messages'), ('check=', None, "don't update the catalog, just return the status. Return code 0 means nothing would change. Return code 1 means that the catalog would be updated"), ('ignore-pot-creation-date=', None, 'ignore changes to POT-Creation-Date when updating or checking')]
    boolean_options = ['omit-header', 'no-wrap', 'ignore-obsolete', 'init-missing', 'no-fuzzy-matching', 'previous', 'update-header-comment', 'check', 'ignore-pot-creation-date']

    def initialize_options(self):
        self.domain = 'messages'
        self.input_file = None
        self.output_dir = None
        self.output_file = None
        self.omit_header = False
        self.locale = None
        self.width = None
        self.no_wrap = False
        self.ignore_obsolete = False
        self.init_missing = False
        self.no_fuzzy_matching = False
        self.update_header_comment = False
        self.previous = False
        self.check = False
        self.ignore_pot_creation_date = False

    def finalize_options(self):
        if not self.input_file:
            raise OptionError('you must specify the input file')
        if not self.output_file and (not self.output_dir):
            raise OptionError('you must specify the output file or directory')
        if self.output_file and (not self.locale):
            raise OptionError('you must specify the locale')
        if self.init_missing:
            if not self.locale:
                raise OptionError('you must specify the locale for the init-missing option to work')
            try:
                self._locale = Locale.parse(self.locale)
            except UnknownLocaleError as e:
                raise OptionError(e) from e
        else:
            self._locale = None
        if self.no_wrap and self.width:
            raise OptionError("'--no-wrap' and '--width' are mutually exclusive")
        if not self.no_wrap and (not self.width):
            self.width = 76
        elif self.width is not None:
            self.width = int(self.width)
        if self.no_fuzzy_matching and self.previous:
            self.previous = False

    def run(self):
        check_status = {}
        po_files = []
        if not self.output_file:
            if self.locale:
                po_files.append((self.locale, os.path.join(self.output_dir, self.locale, 'LC_MESSAGES', f'{self.domain}.po')))
            else:
                for locale in os.listdir(self.output_dir):
                    po_file = os.path.join(self.output_dir, locale, 'LC_MESSAGES', f'{self.domain}.po')
                    if os.path.exists(po_file):
                        po_files.append((locale, po_file))
        else:
            po_files.append((self.locale, self.output_file))
        if not po_files:
            raise OptionError('no message catalogs found')
        domain = self.domain
        if not domain:
            domain = os.path.splitext(os.path.basename(self.input_file))[0]
        with open(self.input_file, 'rb') as infile:
            template = read_po(infile)
        for locale, filename in po_files:
            if self.init_missing and (not os.path.exists(filename)):
                if self.check:
                    check_status[filename] = False
                    continue
                self.log.info('creating catalog %s based on %s', filename, self.input_file)
                with open(self.input_file, 'rb') as infile:
                    catalog = read_po(infile, locale=self.locale)
                catalog.locale = self._locale
                catalog.revision_date = datetime.datetime.now(LOCALTZ)
                catalog.fuzzy = False
                with open(filename, 'wb') as outfile:
                    write_po(outfile, catalog)
            self.log.info('updating catalog %s based on %s', filename, self.input_file)
            with open(filename, 'rb') as infile:
                catalog = read_po(infile, locale=locale, domain=domain)
            catalog.update(template, self.no_fuzzy_matching, update_header_comment=self.update_header_comment, update_creation_date=not self.ignore_pot_creation_date)
            tmpname = os.path.join(os.path.dirname(filename), tempfile.gettempprefix() + os.path.basename(filename))
            try:
                with open(tmpname, 'wb') as tmpfile:
                    write_po(tmpfile, catalog, omit_header=self.omit_header, ignore_obsolete=self.ignore_obsolete, include_previous=self.previous, width=self.width)
            except Exception:
                os.remove(tmpname)
                raise
            if self.check:
                with open(filename, 'rb') as origfile:
                    original_catalog = read_po(origfile)
                with open(tmpname, 'rb') as newfile:
                    updated_catalog = read_po(newfile)
                updated_catalog.revision_date = original_catalog.revision_date
                check_status[filename] = updated_catalog.is_identical(original_catalog)
                os.remove(tmpname)
                continue
            try:
                os.rename(tmpname, filename)
            except OSError:
                os.remove(filename)
                shutil.copy(tmpname, filename)
                os.remove(tmpname)
        if self.check:
            for filename, up_to_date in check_status.items():
                if up_to_date:
                    self.log.info('Catalog %s is up to date.', filename)
                else:
                    self.log.warning('Catalog %s is out of date.', filename)
            if not all(check_status.values()):
                raise BaseError('Some catalogs are out of date.')
            else:
                self.log.info('All the catalogs are up-to-date.')
            return