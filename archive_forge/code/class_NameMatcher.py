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
class NameMatcher(SectionMatcher):

    def __init__(self, store, section_id):
        super().__init__(store)
        self.section_id = section_id

    def match(self, section):
        return section.id == self.section_id