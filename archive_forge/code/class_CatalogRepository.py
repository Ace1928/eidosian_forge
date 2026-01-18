import os
import re
import warnings
from datetime import datetime, timezone
from os import path
from typing import TYPE_CHECKING, Callable, Generator, List, NamedTuple, Optional, Tuple, Union
import babel.dates
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.osutil import SEP, canon_path, relpath
class CatalogRepository:
    """A repository for message catalogs."""

    def __init__(self, basedir: str, locale_dirs: List[str], language: str, encoding: str) -> None:
        self.basedir = basedir
        self._locale_dirs = locale_dirs
        self.language = language
        self.encoding = encoding

    @property
    def locale_dirs(self) -> Generator[str, None, None]:
        if not self.language:
            return
        for locale_dir in self._locale_dirs:
            locale_dir = path.join(self.basedir, locale_dir)
            locale_path = path.join(locale_dir, self.language, 'LC_MESSAGES')
            if path.exists(locale_path):
                yield locale_dir
            else:
                logger.verbose(__('locale_dir %s does not exists'), locale_path)

    @property
    def pofiles(self) -> Generator[Tuple[str, str], None, None]:
        for locale_dir in self.locale_dirs:
            basedir = path.join(locale_dir, self.language, 'LC_MESSAGES')
            for root, dirnames, filenames in os.walk(basedir):
                for dirname in dirnames:
                    if dirname.startswith('.'):
                        dirnames.remove(dirname)
                for filename in filenames:
                    if filename.endswith('.po'):
                        fullpath = path.join(root, filename)
                        yield (basedir, relpath(fullpath, basedir))

    @property
    def catalogs(self) -> Generator[CatalogInfo, None, None]:
        for basedir, filename in self.pofiles:
            domain = canon_path(path.splitext(filename)[0])
            yield CatalogInfo(basedir, domain, self.encoding)