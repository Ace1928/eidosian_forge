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
class ExtractMessages(CommandMixin):
    description = 'extract localizable strings from the project code'
    user_options = [('charset=', None, 'charset to use in the output file (default "utf-8")'), ('keywords=', 'k', 'space-separated list of keywords to look for in addition to the defaults (may be repeated multiple times)'), ('no-default-keywords', None, 'do not include the default keywords'), ('mapping-file=', 'F', 'path to the mapping configuration file'), ('no-location', None, 'do not include location comments with filename and line number'), ('add-location=', None, 'location lines format. If it is not given or "full", it generates the lines with both file name and line number. If it is "file", the line number part is omitted. If it is "never", it completely suppresses the lines (same as --no-location).'), ('omit-header', None, 'do not include msgid "" entry in header'), ('output-file=', 'o', 'name of the output file'), ('width=', 'w', 'set output line width (default 76)'), ('no-wrap', None, 'do not break long message lines, longer than the output line width, into several lines'), ('sort-output', None, 'generate sorted output (default False)'), ('sort-by-file', None, 'sort output by file location (default False)'), ('msgid-bugs-address=', None, 'set report address for msgid'), ('copyright-holder=', None, 'set copyright holder in output'), ('project=', None, 'set project name in output'), ('version=', None, 'set project version in output'), ('add-comments=', 'c', 'place comment block with TAG (or those preceding keyword lines) in output file. Separate multiple TAGs with commas(,)'), ('strip-comments', 's', 'strip the comment TAGs from the comments.'), ('input-paths=', None, 'files or directories that should be scanned for messages. Separate multiple files or directories with commas(,)'), ('input-dirs=', None, 'alias for input-paths (does allow files as well as directories).'), ('ignore-dirs=', None, 'Patterns for directories to ignore when scanning for messages. Separate multiple patterns with spaces (default ".* ._")'), ('header-comment=', None, 'header comment for the catalog'), ('last-translator=', None, 'set the name and email of the last translator in output')]
    boolean_options = ['no-default-keywords', 'no-location', 'omit-header', 'no-wrap', 'sort-output', 'sort-by-file', 'strip-comments']
    as_args = 'input-paths'
    multiple_value_options = ('add-comments', 'keywords', 'ignore-dirs')
    option_aliases = {'keywords': ('--keyword',), 'mapping-file': ('--mapping',), 'output-file': ('--output',), 'strip-comments': ('--strip-comment-tags',), 'last-translator': ('--last-translator',)}
    option_choices = {'add-location': ('full', 'file', 'never')}

    def initialize_options(self):
        self.charset = 'utf-8'
        self.keywords = None
        self.no_default_keywords = False
        self.mapping_file = None
        self.no_location = False
        self.add_location = None
        self.omit_header = False
        self.output_file = None
        self.input_dirs = None
        self.input_paths = None
        self.width = None
        self.no_wrap = False
        self.sort_output = False
        self.sort_by_file = False
        self.msgid_bugs_address = None
        self.copyright_holder = None
        self.project = None
        self.version = None
        self.add_comments = None
        self.strip_comments = False
        self.include_lineno = True
        self.ignore_dirs = None
        self.header_comment = None
        self.last_translator = None

    def finalize_options(self):
        if self.input_dirs:
            if not self.input_paths:
                self.input_paths = self.input_dirs
            else:
                raise OptionError('input-dirs and input-paths are mutually exclusive')
        keywords = {} if self.no_default_keywords else DEFAULT_KEYWORDS.copy()
        keywords.update(parse_keywords(listify_value(self.keywords)))
        self.keywords = keywords
        if not self.keywords:
            raise OptionError('you must specify new keywords if you disable the default ones')
        if not self.output_file:
            raise OptionError('no output file specified')
        if self.no_wrap and self.width:
            raise OptionError("'--no-wrap' and '--width' are mutually exclusive")
        if not self.no_wrap and (not self.width):
            self.width = 76
        elif self.width is not None:
            self.width = int(self.width)
        if self.sort_output and self.sort_by_file:
            raise OptionError("'--sort-output' and '--sort-by-file' are mutually exclusive")
        if self.input_paths:
            if isinstance(self.input_paths, str):
                self.input_paths = re.split(',\\s*', self.input_paths)
        elif self.distribution is not None:
            self.input_paths = dict.fromkeys([k.split('.', 1)[0] for k in self.distribution.packages or ()]).keys()
        else:
            self.input_paths = []
        if not self.input_paths:
            raise OptionError('no input files or directories specified')
        for path in self.input_paths:
            if not os.path.exists(path):
                raise OptionError(f'Input path: {path} does not exist')
        self.add_comments = listify_value(self.add_comments or (), ',')
        if self.distribution:
            if not self.project:
                self.project = self.distribution.get_name()
            if not self.version:
                self.version = self.distribution.get_version()
        if self.add_location == 'never':
            self.no_location = True
        elif self.add_location == 'file':
            self.include_lineno = False
        ignore_dirs = listify_value(self.ignore_dirs)
        if ignore_dirs:
            self.directory_filter = _make_directory_filter(self.ignore_dirs)
        else:
            self.directory_filter = None

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

    def run(self):
        mappings = self._get_mappings()
        with open(self.output_file, 'wb') as outfile:
            catalog = Catalog(project=self.project, version=self.version, msgid_bugs_address=self.msgid_bugs_address, copyright_holder=self.copyright_holder, charset=self.charset, header_comment=self.header_comment or DEFAULT_HEADER, last_translator=self.last_translator)
            for path, method_map, options_map in mappings:
                callback = self._build_callback(path)
                if os.path.isfile(path):
                    current_dir = os.getcwd()
                    extracted = check_and_call_extract_file(path, method_map, options_map, callback, self.keywords, self.add_comments, self.strip_comments, current_dir)
                else:
                    extracted = extract_from_dir(path, method_map, options_map, keywords=self.keywords, comment_tags=self.add_comments, callback=callback, strip_comment_tags=self.strip_comments, directory_filter=self.directory_filter)
                for filename, lineno, message, comments, context in extracted:
                    if os.path.isfile(path):
                        filepath = filename
                    else:
                        filepath = os.path.normpath(os.path.join(path, filename))
                    catalog.add(message, None, [(filepath, lineno)], auto_comments=comments, context=context)
            self.log.info('writing PO template file to %s', self.output_file)
            write_po(outfile, catalog, width=self.width, no_location=self.no_location, omit_header=self.omit_header, sort_output=self.sort_output, sort_by_file=self.sort_by_file, include_lineno=self.include_lineno)

    def _get_mappings(self):
        mappings = []
        if self.mapping_file:
            with open(self.mapping_file) as fileobj:
                method_map, options_map = parse_mapping(fileobj)
            for path in self.input_paths:
                mappings.append((path, method_map, options_map))
        elif getattr(self.distribution, 'message_extractors', None):
            message_extractors = self.distribution.message_extractors
            for path, mapping in message_extractors.items():
                if isinstance(mapping, str):
                    method_map, options_map = parse_mapping(StringIO(mapping))
                else:
                    method_map, options_map = ([], {})
                    for pattern, method, options in mapping:
                        method_map.append((pattern, method))
                        options_map[pattern] = options or {}
                mappings.append((path, method_map, options_map))
        else:
            for path in self.input_paths:
                mappings.append((path, DEFAULT_MAPPING, {}))
        return mappings