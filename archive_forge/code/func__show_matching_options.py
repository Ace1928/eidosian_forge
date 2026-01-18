import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def _show_matching_options(self, name, directory, scope):
    name = lazy_regex.lazy_compile(name)
    name._compile_and_collapse()
    cur_store_id = None
    cur_section = None
    conf = self._get_stack(directory, scope)
    for store, section in conf.iter_sections():
        for oname in section.iter_option_names():
            if name.search(oname):
                if cur_store_id != store.id:
                    self.outf.write('{}:\n'.format(store.id))
                    cur_store_id = store.id
                    cur_section = None
                if section.id is not None and cur_section != section.id:
                    self.outf.write('  [{}]\n'.format(section.id))
                    cur_section = section.id
                value = section.get(oname, expand=False)
                value = self._quote_multiline(value)
                self.outf.write('  {} = {}\n'.format(oname, value))