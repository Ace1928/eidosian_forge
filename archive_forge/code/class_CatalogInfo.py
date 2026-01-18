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
class CatalogInfo(LocaleFileInfoBase):

    @property
    def po_file(self) -> str:
        return self.domain + '.po'

    @property
    def mo_file(self) -> str:
        return self.domain + '.mo'

    @property
    def po_path(self) -> str:
        return path.join(self.base_dir, self.po_file)

    @property
    def mo_path(self) -> str:
        return path.join(self.base_dir, self.mo_file)

    def is_outdated(self) -> bool:
        return not path.exists(self.mo_path) or path.getmtime(self.mo_path) < path.getmtime(self.po_path)

    def write_mo(self, locale: str, use_fuzzy: bool=False) -> None:
        with open(self.po_path, encoding=self.charset) as file_po:
            try:
                po = read_po(file_po, locale)
            except Exception as exc:
                logger.warning(__('reading error: %s, %s'), self.po_path, exc)
                return
        with open(self.mo_path, 'wb') as file_mo:
            try:
                write_mo(file_mo, po, use_fuzzy)
            except Exception as exc:
                logger.warning(__('writing error: %s, %s'), self.mo_path, exc)