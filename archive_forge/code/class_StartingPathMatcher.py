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
class StartingPathMatcher(SectionMatcher):
    """Select sections for a given location respecting the Store order."""

    def __init__(self, store, location):
        super().__init__(store)
        if location.startswith('file://'):
            location = urlutils.local_path_from_url(location)
        self.location = location

    def get_sections(self):
        """Get all sections matching ``location`` in the store.

        The most generic sections are described first in the store, then more
        specific ones can be provided for reduced scopes.

        The returned section are therefore returned in the reversed order so
        the most specific ones can be found first.
        """
        location_parts = self.location.rstrip('/').split('/')
        store = self.store
        for _, section in reversed(list(store.get_sections())):
            if section.id is None:
                yield (store, LocationSection(section, self.location))
                continue
            section_path = section.id
            if section_path.startswith('file://'):
                section_path = urlutils.local_path_from_url(section_path)
            if self.location.startswith(section_path) or fnmatch.fnmatch(self.location, section_path):
                section_parts = section_path.rstrip('/').split('/')
                extra_path = '/'.join(location_parts[len(section_parts):])
                yield (store, LocationSection(section, extra_path))